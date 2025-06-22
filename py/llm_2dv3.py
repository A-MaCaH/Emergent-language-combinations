#!/usr/bin/env python3
"""
Implementation of "Spontaneous Emergence of Agent Individuality Through Social Interactions in LLM-based Communities"
With ASCII board visualization

This simulation demonstrates how homogeneous LLM agents develop unique personalities and behaviors
through social interactions in a 2D environment.
"""

import asyncio
import json
import random
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import requests
from datetime import datetime

# Configuration
GRID_SIZE = 12  # Smaller for better console display
NUM_AGENTS = 6
MAX_STEPS = 2
MESSAGE_RANGE = 3
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
        self.memory = f"I am {agent_id}, exploring this environment."
        self.messages_received: List[Message] = []
        self.messages_sent: List[Message] = []
        self.movement_history: List[str] = []
        self.step_count = 0
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
        
        if nearby_messages:
            messages_text = "\n".join([f"{msg.sender_id}: {msg.content}" 
                                     for msg in nearby_messages])
            self.interaction_count += len(nearby_messages)
        else:
            messages_text = "No nearby messages"
        
        prompt = f"""You are {self.get_current_state()}.

Instructions:
Generate a short message to communicate with nearby agents. Be natural and develop your personality.

Your memory:
{self.memory}

Nearby messages:
{messages_text}

Generate a brief message (1-2 sentences):"""

        response = await LLMInterface.generate_response(prompt, max_tokens=80)
        return response
    
    async def update_memory(self, nearby_messages: List[Message]) -> str:
        """Update memory based on current situation and messages"""
        
        if nearby_messages:
            messages_text = "\n".join([f"{msg.sender_id}: {msg.content}" 
                                     for msg in nearby_messages])
        else:
            messages_text = "No interactions"
        
        prompt = f"""You are {self.get_current_state()}.

Instructions:
Update your memory based on recent experiences. Develop your unique personality.

Current memory:
{self.memory}

Recent interactions:
{messages_text}

Generate updated memory (2-3 sentences):"""

        response = await LLMInterface.generate_response(prompt, max_tokens=120)
        self.memory = response
        return response
    
    async def generate_movement(self) -> str:
        """Generate movement command based on memory and personality"""
        
        prompt = f"""You are {self.get_current_state()}.

Instructions:
Choose your next movement based on your personality.
Options: right, left, up, down, stay

Your memory and personality:
{self.memory}

Position: ({self.position.x}, {self.position.y}) in {GRID_SIZE}x{GRID_SIZE} grid

Choose one: right, left, up, down, or stay"""

        response = await LLMInterface.generate_response(prompt, max_tokens=30)
        
        # Parse movement command
        response_lower = response.lower()
        if "right" in response_lower:
            return "right"
        elif "left" in response_lower:
            return "left"
        elif "up" in response_lower:
            return "up"
        elif "down" in response_lower:
            return "down"
        else:
            return "stay"
    
    def execute_movement(self, command: str) -> None:
        """Execute movement command"""
        if command == "right":
            self.position.x = min(self.position.x + 1, GRID_SIZE - 1)
        elif command == "left":
            self.position.x = max(self.position.x - 1, 0)
        elif command == "up":
            self.position.y = max(self.position.y - 1, 0)  # up decreases y
        elif command == "down":
            self.position.y = min(self.position.y + 1, GRID_SIZE - 1)  # down increases y
        
        self.movement_history.append(command)

class BoardDisplay:
    """ASCII board display for the simulation"""
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.agent_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_board(self, agents: List[Agent], step: int, messages: List[Message]):
        """Display the current board state"""
        self.clear_screen()
        
        print("=" * 80)
        print(f"🤖 LLM AGENT SIMULATION - STEP {step}")
        print("=" * 80)
        
        # Create grid
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place agents on grid
        agent_positions = {}
        for i, agent in enumerate(agents):
            symbol = self.agent_symbols[i] if i < len(self.agent_symbols) else str(i)
            grid[agent.position.y][agent.position.x] = symbol
            agent_positions[symbol] = agent
        
        # Display grid with coordinates
        print("   ", end="")
        for x in range(self.grid_size):
            print(f"{x:2d}", end=" ")
        print()
        
        for y in range(self.grid_size):
            print(f"{y:2d} ", end="")
            for x in range(self.grid_size):
                cell = grid[y][x]
                if cell == '.':
                    print(" .", end=" ")
                else:
                    print(f" {cell}", end=" ")
            print()
        
        print("\n" + "=" * 50)
        print("📍 AGENT POSITIONS:")
        for i, agent in enumerate(agents):
            symbol = self.agent_symbols[i] if i < len(self.agent_symbols) else str(i)
            last_move = agent.movement_history[-1] if agent.movement_history else "none"
            print(f"  {symbol} = {agent.agent_id}: ({agent.position.x},{agent.position.y}) "
                  f"[last: {last_move}, interactions: {agent.interaction_count}]")
        
        print("\n" + "=" * 50)
        print("💬 MESSAGES THIS STEP:")
        if messages:
            for msg in messages[-6:]:  # Show last 6 messages
                # Truncate long messages
                content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                print(f"  {msg.sender_id}: {content}")
        else:
            print("  No messages this step")
        
        print("\n" + "=" * 50)
        print("🧠 AGENT MEMORIES (truncated):")
        for i, agent in enumerate(agents):
            symbol = self.agent_symbols[i] if i < len(self.agent_symbols) else str(i)
            memory_preview = agent.memory[:80] + "..." if len(agent.memory) > 80 else agent.memory
            print(f"  {symbol} {agent.agent_id}: {memory_preview}")
        
        print("\n" + "=" * 80)

class SimulationWithBoard:
    """Main simulation with ASCII board display"""
    
    def __init__(self):
        self.agents: List[Agent] = []
        self.all_messages: List[Message] = []
        self.step = 0
        self.board_display = BoardDisplay(GRID_SIZE)
        self.simulation_log = []
        
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
        
        # Show initial board
        self.board_display.display_board(self.agents, 0, [])
        input("Press Enter to start simulation...")
    
    async def run_step(self) -> None:
        """Execute one simulation step"""
        # Step 1: Generate messages
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
        
        # Step 2: Distribute messages
        self.all_messages.extend(new_messages)
        for agent in self.agents:
            nearby_messages = agent.get_nearby_messages(new_messages)
            agent.messages_received.extend(nearby_messages)
        
        # Step 3: Update memories
        for agent in self.agents:
            nearby_messages = agent.get_nearby_messages(new_messages)
            await agent.update_memory(nearby_messages)
        
        # Step 4: Generate and execute movements
        for agent in self.agents:
            movement_command = await agent.generate_movement()
            agent.execute_movement(movement_command)
        
        # Log step data
        self.log_step_data(new_messages)
        
        # Display updated board
        self.board_display.display_board(self.agents, self.step + 1, new_messages)
        
        self.step += 1
    
    def log_step_data(self, messages: List[Message]) -> None:
        """Log step data for later analysis"""
        step_data = {
            'step': self.step,
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'messages': []
        }
        
        for agent in self.agents:
            step_data['agents'][agent.agent_id] = {
                'position': {'x': agent.position.x, 'y': agent.position.y},
                'memory': agent.memory,
                'interaction_count': agent.interaction_count,
                'last_movement': agent.movement_history[-1] if agent.movement_history else 'none'
            }
        
        for msg in messages:
            step_data['messages'].append({
                'sender': msg.sender_id,
                'content': msg.content,
                'position': {'x': msg.position.x, 'y': msg.position.y}
            })
        
        self.simulation_log.append(step_data)
    
    async def run_simulation(self) -> None:
        """Run the complete simulation"""
        print("Starting LLM Agent Simulation with Board Display")
        
        self.initialize_agents()
        
        for step_num in range(MAX_STEPS):
            try:
                print(f"\nProcessing step {step_num + 1}...")
                await self.run_step()
                
                # Pause between steps
                if step_num < MAX_STEPS - 1:  # Don't pause on last step
                    input(f"Step {step_num + 1} completed. Press Enter for next step...")
                
            except KeyboardInterrupt:
                print("\nSimulation interrupted by user")
                break
            except Exception as e:
                print(f"Error in step {step_num}: {e}")
                break
        
        print("\n🎉 Simulation completed!")
        self.save_simulation_log()
        
        # Final analysis
        self.show_final_analysis()
    
    def show_final_analysis(self) -> None:
        """Show final analysis of the simulation"""
        print("\n" + "=" * 80)
        print("📊 FINAL ANALYSIS")
        print("=" * 80)
        
        # Movement patterns
        print("\n🚶 MOVEMENT PATTERNS:")
        for agent in self.agents:
            if agent.movement_history:
                move_counts = {}
                for move in agent.movement_history:
                    move_counts[move] = move_counts.get(move, 0) + 1
                
                most_common = max(move_counts.items(), key=lambda x: x[1])
                print(f"  {agent.agent_id}: Most common move = {most_common[0]} "
                      f"({most_common[1]}/{len(agent.movement_history)} times)")
        
        # Final positions and clustering
        print("\n🎯 FINAL POSITIONS:")
        clusters = self.find_simple_clusters()
        for i, cluster in enumerate(clusters):
            agents_in_cluster = [f"{agent.agent_id}({agent.position.x},{agent.position.y})" 
                               for agent in cluster]
            print(f"  Cluster {i+1}: {', '.join(agents_in_cluster)}")
        
        # Interaction summary
        print("\n💬 INTERACTION SUMMARY:")
        total_interactions = sum(agent.interaction_count for agent in self.agents)
        print(f"  Total interactions: {total_interactions}")
        for agent in self.agents:
            print(f"  {agent.agent_id}: {agent.interaction_count} interactions")
    
    def find_simple_clusters(self, max_distance: int = 2) -> List[List[Agent]]:
        """Find clusters of agents based on proximity"""
        clusters = []
        visited = set()
        
        for agent in self.agents:
            if agent.agent_id in visited:
                continue
            
            cluster = [agent]
            visited.add(agent.agent_id)
            
            for other_agent in self.agents:
                if (other_agent.agent_id not in visited and 
                    agent.position.distance_to(other_agent.position) <= max_distance):
                    cluster.append(other_agent)
                    visited.add(other_agent.agent_id)
            
            clusters.append(cluster)
        
        return clusters
    
    def save_simulation_log(self) -> None:
        """Save simulation log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_log_{timestamp}.json"
        
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(self.simulation_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Simulation log saved to: {filename}")

# Main execution
async def main():
    """Main function to run the simulation"""
    print("🤖 Enhanced LLM Agent Simulation with ASCII Board")
    print("=" * 60)
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            if MODEL_NAME in model_names or any(MODEL_NAME in name for name in model_names):
                print(f"✅ Model {MODEL_NAME} is available")
            else:
                print(f"❌ Model {MODEL_NAME} not found. Available: {model_names}")
                print("Run: ollama pull llama3")
                return
        else:
            print("❌ Ollama not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to Ollama. Start with: ollama serve")
        return
    
    # Run simulation
    simulation = SimulationWithBoard()
    await simulation.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())