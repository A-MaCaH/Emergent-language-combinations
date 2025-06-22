#!/usr/bin/env python3
"""
Implementation of "Spontaneous Emergence of Agent Individuality Through Social Interactions in LLM-based Communities"
Enhanced version with real-time visualization and detailed logging

This simulation demonstrates how homogeneous LLM agents develop unique personalities and behaviors
through social interactions in a 2D environment.
"""

import asyncio
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import requests
import re
from datetime import datetime
import pandas as pd
import time
import os

# Configuration
GRID_SIZE = 20  # Reduced for better visualization
NUM_AGENTS = 8  # Reduced for clarity
MAX_STEPS = 5  # Reduced for demo
MESSAGE_RANGE = 5  # Chebyshev distance for message exchange
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
VISUALIZATION_DELAY = 2.0  # Seconds between steps for visualization

@dataclass
class Position:
    """Represents a 2D position in the grid"""
    x: int
    y: int
    
    def distance_to(self, other: 'Position') -> int:
        """Calculate Chebyshev distance to another position"""
        return max(abs(self.x - other.x), abs(self.y - other.y))
    
    def is_valid(self) -> bool:
        """Check if position is within grid bounds"""
        return 0 <= self.x < GRID_SIZE and 0 <= self.y < GRID_SIZE

@dataclass
class Message:
    """Represents a message between agents"""
    sender_id: str
    content: str
    step: int
    position: Position

class LLMInterface:
    """Interface for communicating with Ollama"""
    
    @staticmethod
    async def generate_response(prompt: str, max_tokens: int = 256) -> str:
        """Generate response using Ollama API"""
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40
                }
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"

class Agent:
    """LLM-based agent with memory, messaging, and movement capabilities"""
    
    def __init__(self, agent_id: str, initial_position: Position):
        self.agent_id = agent_id
        self.position = initial_position
        self.memory = "I am a newly created agent exploring this environment."
        self.messages_received: List[Message] = []
        self.messages_sent: List[Message] = []
        self.movement_history: List[str] = []
        self.step_count = 0
        self.personality_traits = []  # Will emerge through interactions
        self.interaction_count = 0
        
    def get_current_state(self) -> str:
        """Get current state description for prompts"""
        return f"Agent {self.agent_id} at position ({self.position.x}, {self.position.y})"
    
    def get_nearby_messages(self, all_messages: List[Message]) -> List[Message]:
        """Get messages from agents within communication range"""
        nearby_messages = []
        for msg in all_messages:
            if (msg.sender_id != self.agent_id and 
                msg.step == self.step_count and
                self.position.distance_to(msg.position) <= MESSAGE_RANGE):
                nearby_messages.append(msg)
        return nearby_messages
    
    async def generate_message(self, nearby_messages: List[Message]) -> str:
        """Generate a message based on current state and nearby messages"""
        
        # Format nearby messages
        if nearby_messages:
            messages_text = "\n".join([f"Agent {msg.sender_id}: {msg.content}" 
                                     for msg in nearby_messages])
            self.interaction_count += len(nearby_messages)
        else:
            messages_text = "No messages"
        
        prompt = f"""You are {self.get_current_state()}.

Current state of each agent itself:
{self.get_current_state()}

Instructions:
Generate a short message to communicate with nearby agents. Be natural and conversational.
Develop your own personality through these interactions.

Agent's own memory:
[{self.memory}]

Messages from nearby agents:
[{messages_text}]

Generate a brief, natural message (1-2 sentences):"""

        response = await LLMInterface.generate_response(prompt, max_tokens=100)
        return response
    
    async def update_memory(self, nearby_messages: List[Message]) -> str:
        """Update memory based on current situation and messages"""
        
        if nearby_messages:
            messages_text = "\n".join([f"Agent {msg.sender_id}: {msg.content}" 
                                     for msg in nearby_messages])
        else:
            messages_text = "No messages"
        
        prompt = f"""You are {self.get_current_state()}.

Instructions:
Update your memory based on recent experiences and interactions. 
Develop your personality and preferences based on what you've experienced.

Current memory:
[{self.memory}]

Recent messages from nearby agents:
[{messages_text}]

Step: {self.step_count}
Interactions so far: {self.interaction_count}

Generate an updated memory summary (2-3 sentences):"""

        response = await LLMInterface.generate_response(prompt, max_tokens=150)
        self.memory = response
        return response
    
    async def generate_movement(self) -> str:
        """Generate movement command based on memory and personality"""
        
        prompt = f"""You are {self.get_current_state()}.

Instructions:
Choose your next movement based on your personality and memory.
You can move: right (x+1), left (x-1), up (y+1), down (y-1), or stay in place (stay).

Your memory and personality:
[{self.memory}]

Current position: ({self.position.x}, {self.position.y})
Grid size: {GRID_SIZE}x{GRID_SIZE}

Choose one movement: x+1, x-1, y+1, y-1, or stay"""

        response = await LLMInterface.generate_response(prompt, max_tokens=50)
        
        # Parse movement command
        response_lower = response.lower()
        if "x+1" in response_lower or "right" in response_lower:
            return "x+1"
        elif "x-1" in response_lower or "left" in response_lower:
            return "x-1"
        elif "y+1" in response_lower or "up" in response_lower:
            return "y+1"
        elif "y-1" in response_lower or "down" in response_lower:
            return "y-1"
        else:
            return "stay"
    
    def execute_movement(self, command: str) -> None:
        """Execute movement command"""
        old_position = Position(self.position.x, self.position.y)
        
        if command == "x+1":
            self.position.x = min(self.position.x + 1, GRID_SIZE - 1)
        elif command == "x-1":
            self.position.x = max(self.position.x - 1, 0)
        elif command == "y+1":
            self.position.y = min(self.position.y + 1, GRID_SIZE - 1)
        elif command == "y-1":
            self.position.y = max(self.position.y - 1, 0)
        # "stay" doesn't change position
        
        self.movement_history.append(command)

class RealTimeVisualizer:
    """Real-time visualization of the simulation"""
    
    def __init__(self, grid_size: int, agents: List[Agent]):
        self.grid_size = grid_size
        self.agents = agents
        self.fig, (self.ax_grid, self.ax_messages) = plt.subplots(1, 2, figsize=(16, 8))
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
        self.agent_circles = {}
        self.message_text = []
        
        # Setup grid plot
        self.ax_grid.set_xlim(-0.5, grid_size - 0.5)
        self.ax_grid.set_ylim(-0.5, grid_size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.grid(True, alpha=0.3)
        self.ax_grid.set_title('Agent Positions and Movement', fontsize=14, fontweight='bold')
        self.ax_grid.set_xlabel('X Position')
        self.ax_grid.set_ylabel('Y Position')
        
        # Setup message plot
        self.ax_messages.set_xlim(0, 1)
        self.ax_messages.set_ylim(0, 1)
        self.ax_messages.axis('off')
        self.ax_messages.set_title('Agent Messages', fontsize=14, fontweight='bold')
        
        # Initialize agent circles
        for i, agent in enumerate(self.agents):
            circle = Circle((agent.position.x, agent.position.y), 0.3, 
                          color=self.colors[i], alpha=0.7, label=agent.agent_id)
            self.ax_grid.add_patch(circle)
            self.agent_circles[agent.agent_id] = circle
            
            # Add agent label
            self.ax_grid.annotate(agent.agent_id, 
                                (agent.position.x, agent.position.y),
                                ha='center', va='center', fontweight='bold', fontsize=8)
        
        self.ax_grid.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
    def update_visualization(self, step: int, messages: List[Message]):
        """Update the visualization with current step data"""
        # Clear previous annotations and messages
        for txt in self.ax_grid.texts[len(self.agents):]:  # Keep agent labels
            txt.remove()
        self.ax_messages.clear()
        self.ax_messages.set_xlim(0, 1)
        self.ax_messages.set_ylim(0, 1)
        self.ax_messages.axis('off')
        self.ax_messages.set_title(f'Agent Messages - Step {step}', fontsize=14, fontweight='bold')
        
        # Update agent positions
        for i, agent in enumerate(self.agents):
            circle = self.agent_circles[agent.agent_id]
            circle.center = (agent.position.x, agent.position.y)
            
            # Add agent label at new position
            self.ax_grid.annotate(agent.agent_id, 
                                (agent.position.x, agent.position.y),
                                ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Display messages
        y_pos = 0.95
        for i, msg in enumerate(messages):
            if i >= 12:  # Limit number of messages displayed
                break
            agent_color = self.colors[int(msg.sender_id.replace('agent', ''))]
            
            # Truncate long messages
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            
            self.ax_messages.text(0.02, y_pos, f"{msg.sender_id}:", 
                                fontweight='bold', color=agent_color, fontsize=10)
            self.ax_messages.text(0.02, y_pos - 0.03, content, 
                                fontsize=9, wrap=True)
            y_pos -= 0.08
        
        # Update title with step information
        self.ax_grid.set_title(f'Agent Positions - Step {step}', fontsize=14, fontweight='bold')
        
        plt.draw()
        plt.pause(0.1)

class EnhancedSimulation:
    """Enhanced simulation environment with real-time visualization"""
    
    def __init__(self):
        self.agents: List[Agent] = []
        self.all_messages: List[Message] = []
        self.step = 0
        self.results = {
            'movements': defaultdict(list),
            'messages': defaultdict(list),
            'memories': defaultdict(list),
            'positions': defaultdict(list),
            'interactions': defaultdict(list)
        }
        self.visualizer = None
        self.data_log = []
        
    def initialize_agents(self) -> None:
        """Initialize agents with random positions"""
        print("Initializing agents...")
        for i in range(NUM_AGENTS):
            initial_pos = Position(
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1)
            )
            agent = Agent(f"agent{i}", initial_pos)
            self.agents.append(agent)
            print(f"Agent {i} initialized at position ({initial_pos.x}, {initial_pos.y})")
        
        # Initialize visualizer
        self.visualizer = RealTimeVisualizer(GRID_SIZE, self.agents)
        
    async def run_step(self) -> None:
        """Execute one simulation step with visualization"""
        print(f"\n--- Step {self.step} ---")
        
        # Step 1: All agents generate messages
        print("Phase 1: Generating messages...")
        new_messages = []
        for agent in self.agents:
            agent.step_count = self.step
            nearby_messages = agent.get_nearby_messages(self.all_messages)
            message_content = await agent.generate_message(nearby_messages)
            
            message = Message(
                sender_id=agent.agent_id,
                content=message_content,
                step=self.step,
                position=Position(agent.position.x, agent.position.y)
            )
            new_messages.append(message)
            agent.messages_sent.append(message)
            print(f"  {agent.agent_id}: {message_content[:50]}...")
        
        # Step 2: Distribute messages
        print("Phase 2: Distributing messages...")
        self.all_messages.extend(new_messages)
        for agent in self.agents:
            nearby_messages = agent.get_nearby_messages(new_messages)
            agent.messages_received.extend(nearby_messages)
        
        # Step 3: Update memories
        print("Phase 3: Updating memories...")
        for agent in self.agents:
            nearby_messages = agent.get_nearby_messages(new_messages)
            await agent.update_memory(nearby_messages)
            print(f"  {agent.agent_id} memory updated")
        
        # Step 4: Generate and execute movements
        print("Phase 4: Processing movements...")
        for agent in self.agents:
            movement_command = await agent.generate_movement()
            old_pos = (agent.position.x, agent.position.y)
            agent.execute_movement(movement_command)
            new_pos = (agent.position.x, agent.position.y)
            print(f"  {agent.agent_id}: {movement_command} {old_pos} -> {new_pos}")
        
        # Record results and log data
        self.record_step_results()
        self.log_step_data(new_messages)
        
        # Update visualization
        if self.visualizer:
            self.visualizer.update_visualization(self.step, new_messages)
            
        self.step += 1
        
        # Wait for visualization
        await asyncio.sleep(VISUALIZATION_DELAY)
    
    def record_step_results(self) -> None:
        """Record results for current step"""
        for agent in self.agents:
            self.results['positions'][agent.agent_id].append(
                (self.step, agent.position.x, agent.position.y))
            self.results['memories'][agent.agent_id].append(
                (self.step, agent.memory))
            self.results['interactions'][agent.agent_id].append(
                (self.step, agent.interaction_count))
            if agent.movement_history:
                self.results['movements'][agent.agent_id].append(
                    (self.step, agent.movement_history[-1]))
        
        step_messages = [msg for msg in self.all_messages if msg.step == self.step]
        for msg in step_messages:
            self.results['messages'][msg.sender_id].append((self.step, msg.content))
    
    def log_step_data(self, messages: List[Message]) -> None:
        """Log detailed step data for analysis"""
        step_data = {
            'step': self.step,
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'messages': []
        }
        
        # Log agent data
        for agent in self.agents:
            step_data['agents'][agent.agent_id] = {
                'position': {'x': agent.position.x, 'y': agent.position.y},
                'memory': agent.memory,
                'interaction_count': agent.interaction_count,
                'last_movement': agent.movement_history[-1] if agent.movement_history else 'none'
            }
        
        # Log messages
        for msg in messages:
            step_data['messages'].append({
                'sender': msg.sender_id,
                'content': msg.content,
                'position': {'x': msg.position.x, 'y': msg.position.y}
            })
        
        self.data_log.append(step_data)
    
    async def run_simulation(self) -> None:
        """Run the complete simulation"""
        print("Starting Enhanced LLM Agent Simulation")
        print(f"Configuration: {NUM_AGENTS} agents, {MAX_STEPS} steps, {GRID_SIZE}x{GRID_SIZE} grid")
        
        self.initialize_agents()
        
        # Show initial state
        if self.visualizer:
            self.visualizer.update_visualization(0, [])
            await asyncio.sleep(2)
        
        for step_num in range(MAX_STEPS):
            try:
                await self.run_step()
                
                # Progress update
                if (step_num + 1) % 5 == 0:
                    print(f"\nProgress: {step_num + 1}/{MAX_STEPS} steps completed")
                    self.print_agent_status()
                    
            except KeyboardInterrupt:
                print("\nSimulation interrupted by user")
                break
            except Exception as e:
                print(f"Error in step {step_num}: {e}")
                break
        
        print("\nSimulation completed!")
        self.analyze_results()
        self.save_detailed_logs()
        
        # Keep visualization open
        input("Press Enter to close visualization...")
    
    def print_agent_status(self) -> None:
        """Print current agent status"""
        print("Current agent status:")
        for agent in self.agents:
            print(f"  {agent.agent_id}: pos({agent.position.x}, {agent.position.y}), "
                  f"interactions: {agent.interaction_count}")
    
    def analyze_results(self) -> None:
        """Analyze simulation results with focus on individuality emergence"""
        print("\n=== DETAILED SIMULATION ANALYSIS ===")
        
        # 1. Movement Pattern Analysis
        print("\n1. Movement Pattern Individuality:")
        movement_patterns = {}
        for agent_id, movements in self.results['movements'].items():
            agent_movements = [move[1] for move in movements]
            pattern = {}
            for move in ['x+1', 'x-1', 'y+1', 'y-1', 'stay']:
                pattern[move] = agent_movements.count(move) / len(agent_movements) if agent_movements else 0
            movement_patterns[agent_id] = pattern
            
            print(f"  {agent_id}:")
            for move, freq in pattern.items():
                print(f"    {move}: {freq:.2%}")
        
        # 2. Interaction Analysis
        print("\n2. Social Interaction Patterns:")
        for agent_id, interactions in self.results['interactions'].items():
            if interactions:
                final_interactions = interactions[-1][1]  # Last interaction count
                print(f"  {agent_id}: {final_interactions} total interactions")
        
        # 3. Memory Evolution Analysis
        print("\n3. Memory/Personality Evolution:")
        for agent_id, memories in self.results['memories'].items():
            if len(memories) >= 2:
                initial_memory = memories[0][1]
                final_memory = memories[-1][1]
                print(f"  {agent_id}:")
                print(f"    Initial: {initial_memory[:100]}...")
                print(f"    Final: {final_memory[:100]}...")
        
        # 4. Spatial Clustering
        print("\n4. Spatial Clustering Analysis:")
        final_positions = {}
        for agent in self.agents:
            final_positions[agent.agent_id] = (agent.position.x, agent.position.y)
        
        clusters = self.find_clusters(final_positions)
        print(f"  Number of clusters formed: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            positions_str = [f"{agent}({final_positions[agent][0]},{final_positions[agent][1]})" 
                           for agent in cluster]
            print(f"  Cluster {i + 1}: {', '.join(positions_str)}")
    
    def find_clusters(self, positions: Dict[str, Tuple[int, int]], max_distance: int = 3) -> List[List[str]]:
        """Simple clustering based on position proximity"""
        agents = list(positions.keys())
        clusters = []
        visited = set()
        
        for agent in agents:
            if agent in visited:
                continue
            
            cluster = [agent]
            visited.add(agent)
            queue = [agent]
            
            while queue:
                current = queue.pop(0)
                current_pos = positions[current]
                
                for other_agent in agents:
                    if other_agent in visited:
                        continue
                    
                    other_pos = positions[other_agent]
                    distance = max(abs(current_pos[0] - other_pos[0]), 
                                 abs(current_pos[1] - other_pos[1]))
                    
                    if distance <= max_distance:
                        cluster.append(other_agent)
                        visited.add(other_agent)
                        queue.append(other_agent)
            
            clusters.append(cluster)
        
        return clusters
    
    def save_detailed_logs(self) -> None:
        """Save detailed simulation logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save step-by-step log
        log_filename = f"simulation_detailed_log_{timestamp}.json"
        with open(log_filename, "w", encoding='utf-8') as f:
            json.dump(self.data_log, f, indent=2, ensure_ascii=False)
        
        # Save summary results
        summary_filename = f"simulation_summary_{timestamp}.json"
        with open(summary_filename, "w", encoding='utf-8') as f:
            summary = {
                'configuration': {
                    'grid_size': GRID_SIZE,
                    'num_agents': NUM_AGENTS,
                    'max_steps': MAX_STEPS,
                    'message_range': MESSAGE_RANGE
                },
                'results': dict(self.results)
            }
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed logs saved:")
        print(f"  - Step-by-step: {log_filename}")
        print(f"  - Summary: {summary_filename}")

# Main execution
async def main():
    """Main function to run the enhanced simulation"""
    print("Enhanced LLM Agent Individuality Emergence Simulation")
    print("=" * 60)
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            if MODEL_NAME in model_names or any(MODEL_NAME in name for name in model_names):
                print(f"✓ Model {MODEL_NAME} is available")
            else:
                print(f"⚠ Model {MODEL_NAME} not found. Available models: {model_names}")
                print("Please run: ollama pull llama3")
                return
        else:
            print("✗ Ollama is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("✗ Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434")
        print("Start Ollama with: ollama serve")
        return
    
    # Run enhanced simulation
    simulation = EnhancedSimulation()
    await simulation.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())