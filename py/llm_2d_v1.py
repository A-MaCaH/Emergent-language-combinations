#!/usr/bin/env python3
"""
Implementation of "Spontaneous Emergence of Agent Individuality Through Social Interactions in LLM-based Communities"
Using Ollama and Llama3

This simulation demonstrates how homogeneous LLM agents develop unique personalities and behaviors
through social interactions in a 2D environment.
"""

import asyncio
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import requests
import re
from datetime import datetime
import pandas as pd

# Configuration
GRID_SIZE = 50
NUM_AGENTS = 10
MAX_STEPS = 10
MESSAGE_RANGE = 5  # Chebyshev distance for message exchange
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

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
        self.memory = "No memory"
        self.messages_received: List[Message] = []
        self.messages_sent: List[Message] = []
        self.movement_history: List[str] = []
        self.step_count = 0
        
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
        else:
            messages_text = "No messages"
        
        prompt = f"""You are {self.get_current_state()}.

Current state of each agent itself:
{self.get_current_state()}

Instructions:
Generate a short message to communicate with nearby agents. Be natural and conversational.

Agent's own memory:
[{self.memory}]

All messages received from the surroundings:
[{messages_text}]

Generate a brief message (1-2 sentences):"""

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

Current state of each agent itself:
{self.get_current_state()}

Instructions:
Generate a brief summary of your current situation and recent activities. This will be your memory.

Agent's own memory:
[{self.memory}]

All messages received from the surroundings:
[{messages_text}]

Generate a concise situational summary:"""

        response = await LLMInterface.generate_response(prompt, max_tokens=150)
        self.memory = response
        return response
    
    async def generate_movement(self) -> str:
        """Generate movement command based on memory"""
        
        prompt = f"""You are {self.get_current_state()}.

Current state of each agent itself:
{self.get_current_state()}

Instructions:
Choose your next movement. You can move right (x+1), left (x-1), up (y+1), down (y-1), or stay in place (stay).

Agent's own memory:
[{self.memory}]

Choose one movement command: x+1, x-1, y+1, y-1, or stay"""

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
        new_position = Position(self.position.x, self.position.y)
        
        if command == "x+1":
            new_position.x = (new_position.x + 1) % GRID_SIZE
        elif command == "x-1":
            new_position.x = (new_position.x - 1) % GRID_SIZE
        elif command == "y+1":
            new_position.y = (new_position.y + 1) % GRID_SIZE
        elif command == "y-1":
            new_position.y = (new_position.y - 1) % GRID_SIZE
        # "stay" doesn't change position
        
        self.position = new_position
        self.movement_history.append(command)

class Simulation:
    """Main simulation environment"""
    
    def __init__(self):
        self.agents: List[Agent] = []
        self.all_messages: List[Message] = []
        self.step = 0
        self.results = {
            'movements': defaultdict(list),
            'messages': defaultdict(list),
            'memories': defaultdict(list),
            'positions': defaultdict(list)
        }
        
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
    
    async def run_step(self) -> None:
        """Execute one simulation step"""
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
            agent.execute_movement(movement_command)
            print(f"  {agent.agent_id}: {movement_command} -> ({agent.position.x}, {agent.position.y})")
        
        # Record results
        self.record_step_results()
        self.step += 1
    
    def record_step_results(self) -> None:
        """Record results for current step"""
        for agent in self.agents:
            self.results['positions'][agent.agent_id].append((self.step, agent.position.x, agent.position.y))
            self.results['memories'][agent.agent_id].append((self.step, agent.memory))
            if agent.movement_history:
                self.results['movements'][agent.agent_id].append((self.step, agent.movement_history[-1]))
        
        step_messages = [msg for msg in self.all_messages if msg.step == self.step]
        for msg in step_messages:
            self.results['messages'][msg.sender_id].append((self.step, msg.content))
    
    async def run_simulation(self) -> None:
        """Run the complete simulation"""
        print("Starting LLM Agent Simulation")
        print(f"Configuration: {NUM_AGENTS} agents, {MAX_STEPS} steps, {GRID_SIZE}x{GRID_SIZE} grid")
        
        self.initialize_agents()
        
        for step_num in range(MAX_STEPS):
            try:
                await self.run_step()
                
                # Progress update
                if (step_num + 1) % 10 == 0:
                    print(f"\nProgress: {step_num + 1}/{MAX_STEPS} steps completed")
                    self.print_agent_positions()
                    
            except Exception as e:
                print(f"Error in step {step_num}: {e}")
                break
        
        print("\nSimulation completed!")
        self.analyze_results()
    
    def print_agent_positions(self) -> None:
        """Print current agent positions"""
        print("Current agent positions:")
        for agent in self.agents:
            print(f"  {agent.agent_id}: ({agent.position.x}, {agent.position.y})")
    
    def analyze_results(self) -> None:
        """Analyze and visualize simulation results"""
        print("\n=== SIMULATION ANALYSIS ===")
        
        # Movement analysis
        print("\n1. Movement Command Distribution:")
        all_movements = []
        for agent_id, movements in self.results['movements'].items():
            agent_movements = [move[1] for move in movements]
            all_movements.extend(agent_movements)
        
        movement_counts = {}
        for move in all_movements:
            movement_counts[move] = movement_counts.get(move, 0) + 1
        
        for move, count in sorted(movement_counts.items()):
            percentage = (count / len(all_movements)) * 100
            print(f"  {move}: {count} ({percentage:.1f}%)")
        
        # Message analysis
        print("\n2. Message Analysis:")
        all_messages_content = []
        for agent_id, messages in self.results['messages'].items():
            for step, content in messages:
                all_messages_content.append(content)
        
        print(f"  Total messages generated: {len(all_messages_content)}")
        
        # Look for hashtags and potential hallucinations
        hashtags = set()
        potential_hallucinations = set()
        
        for content in all_messages_content:
            # Find hashtags
            hashtag_matches = re.findall(r'#\w+', content.lower())
            hashtags.update(hashtag_matches)
            
            # Simple heuristic for potential hallucinations (mentions of objects/places)
            hallucination_words = ['tree', 'cave', 'hill', 'treasure', 'forest', 'mountain', 'river', 'building']
            for word in hallucination_words:
                if word in content.lower():
                    potential_hallucinations.add(word)
        
        if hashtags:
            print(f"  Hashtags found: {', '.join(sorted(hashtags))}")
        if potential_hallucinations:
            print(f"  Potential hallucinations: {', '.join(sorted(potential_hallucinations))}")
        
        # Agent clustering analysis
        print("\n3. Agent Clustering Analysis:")
        final_positions = {}
        for agent in self.agents:
            final_positions[agent.agent_id] = (agent.position.x, agent.position.y)
        
        # Simple clustering based on final positions
        clusters = self.find_clusters(final_positions)
        print(f"  Number of clusters formed: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i + 1}: {', '.join(cluster)}")
        
        # Save results
        self.save_results()
    
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
    
    def save_results(self) -> None:
        """Save simulation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        with open(f"simulation_results_{timestamp}.json", "w") as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in self.results.items():
                json_results[key] = {agent_id: data for agent_id, data in value.items()}
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to simulation_results_{timestamp}.json")
        
        # Create visualization
        self.create_visualization(timestamp)
    
    def create_visualization(self, timestamp: str) -> None:
        """Create basic visualization of results"""
        try:
            # Agent trajectories
            plt.figure(figsize=(12, 10))
            
            # Plot 1: Agent trajectories
            plt.subplot(2, 2, 1)
            colors = plt.cm.tab10(np.linspace(0, 1, NUM_AGENTS))
            
            for i, agent in enumerate(self.agents):
                positions = self.results['positions'][agent.agent_id]
                if positions:
                    x_coords = [pos[1] for pos in positions]  # pos[1] is x coordinate
                    y_coords = [pos[2] for pos in positions]  # pos[2] is y coordinate
                    
                    plt.plot(x_coords, y_coords, color=colors[i], alpha=0.7, label=agent.agent_id)
                    plt.scatter(x_coords[0], y_coords[0], color=colors[i], s=100, marker='o')  # Start
                    plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], s=100, marker='s')  # End
            
            plt.xlim(0, GRID_SIZE)
            plt.ylim(0, GRID_SIZE)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Agent Trajectories')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Movement distribution
            plt.subplot(2, 2, 2)
            all_movements = []
            for agent_id, movements in self.results['movements'].items():
                agent_movements = [move[1] for move in movements]
                all_movements.extend(agent_movements)
            
            movement_counts = {}
            for move in all_movements:
                movement_counts[move] = movement_counts.get(move, 0) + 1
            
            moves = list(movement_counts.keys())
            counts = list(movement_counts.values())
            
            plt.bar(moves, counts)
            plt.xlabel('Movement Command')
            plt.ylabel('Frequency')
            plt.title('Movement Command Distribution')
            plt.xticks(rotation=45)
            
            # Plot 3: Agent final positions
            plt.subplot(2, 2, 3)
            for i, agent in enumerate(self.agents):
                plt.scatter(agent.position.x, agent.position.y, 
                          color=colors[i], s=200, alpha=0.7, label=agent.agent_id)
                plt.annotate(agent.agent_id, (agent.position.x, agent.position.y), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlim(0, GRID_SIZE)
            plt.ylim(0, GRID_SIZE)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Final Agent Positions')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Message count per agent
            plt.subplot(2, 2, 4)
            agent_ids = []
            message_counts = []
            
            for agent_id, messages in self.results['messages'].items():
                agent_ids.append(agent_id)
                message_counts.append(len(messages))
            
            plt.bar(agent_ids, message_counts)
            plt.xlabel('Agent ID')
            plt.ylabel('Number of Messages')
            plt.title('Messages Generated per Agent')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"simulation_visualization_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Visualization saved to simulation_visualization_{timestamp}.png")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

# Main execution
async def main():
    """Main function to run the simulation"""
    print("LLM Agent Individuality Emergence Simulation")
    print("=" * 50)
    
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
    
    # Run simulation
    simulation = Simulation()
    await simulation.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())