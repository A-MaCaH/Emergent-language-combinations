#!/usr/bin/env python3
"""
Implementación del Juego de Nombres (Naming Game) usando Ollama y Llama3
Basado en el artículo sobre emergencia de lenguajes en LLMs
"""

import json
import random
import requests
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import statistics

class OllamaAgent:
    """Agente que usa Ollama para tomar decisiones en el juego de nombres"""
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.7, 
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.memory = []  # Historial de interacciones
        self.memory_size = 5  # Tamaño de memoria como en el artículo
        self.score = 0
        self.agent_id = f"agent_{random.randint(1000, 9999)}"
        
    def _make_ollama_request(self, prompt: str) -> str:
        """Hace una petición a Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_k": 40,  # K-sampling como en el artículo
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error en petición a Ollama: {e}")
            return "F"  # Fallback
    
    def _build_system_prompt(self, word_pool: List[str], context_object: str = None) -> str:
        """Construye el prompt del sistema basado en el artículo"""
        memory_str = ""
        if self.memory:
            memory_str = "This is the history of choices in past rounds:\n"
            for i, interaction in enumerate(self.memory[-self.memory_size:], 1):
                memory_str += f"Round {i}: Player 1: {interaction['player1_choice']}, "
                memory_str += f"Player 2: {interaction['player2_choice']}, "
                memory_str += f"Payoff: {interaction['payoff']}\n"
        
        # Contexto del objeto si se proporciona
        object_context = ""
        if context_object:
            object_context = f"""
SCENARIO: You and another player are looking at {context_object}. You both need to agree on what to call it to succeed in this coordination task. 
"""
        
        system_prompt = f"""{object_context}Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds. At each round, Player 1 and Player 2 simultaneously pick a name from the following options: {word_pool}. The payoff that both players get is determined by the following rule:

1. If Players choose the SAME name as each other, they will both be REWARDED with payoff +100 points.
2. If Players choose DIFFERENT names from each other, they will both be PUNISHED with payoff -50 points.

The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player.

{memory_str}

It is now round {len(self.memory) + 1}. The current score of Player 1 is {self.score}. Answer saying which name Player 1 should pick. Please think step by step before making a decision. Remember, examining history explicitly is important. 

Write your answer using the following format: {{'value': <NAME_CHOSEN_BY_PLAYER_1>, 'reason': <YOUR_REASON>}}.

IMPORTANT: You must choose exactly one name from {word_pool}. Answer with the JSON format only."""

        return system_prompt
    
    def choose_action(self, word_pool: List[str], context_object: str = None) -> str:
        """Elige una acción basada en el historial y el pool de palabras"""
        if not self.memory:
            # Primera ronda: elección aleatoria del pool
            return random.choice(word_pool)
        
        system_prompt = self._build_system_prompt(word_pool, context_object)
        user_prompt = "Answer saying which name Player 1 should choose."
        
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        
        response = self._make_ollama_request(full_prompt)
        
        # Extraer la decisión del JSON
        try:
            # Buscar JSON en la respuesta
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                choice = parsed.get('value', '').strip()
                
                # Verificar que la elección esté en el pool
                if choice in word_pool:
                    return choice
        except:
            pass
        
        # Fallback: buscar palabras del pool en la respuesta
        for word in word_pool:
            if word in response.upper():
                return word
        
        # Último fallback: elección aleatoria
        return random.choice(word_pool)
    
    def update_memory(self, player1_choice: str, player2_choice: str, payoff: int):
        """Actualiza la memoria del agente"""
        self.memory.append({
            'player1_choice': player1_choice,
            'player2_choice': player2_choice,
            'payoff': payoff
        })
        self.score += payoff
        
        # Mantener solo las últimas memory_size interacciones
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

class NamingGameSimulation:
    """Simulación del juego de nombres"""
    
    def __init__(self, n_agents: int = 10, word_pool: List[str] = None, 
                 model_name: str = "llama3"):
        self.n_agents = n_agents
        self.word_pool = word_pool or ['F', 'J']  # Pool por defecto del artículo
        self.agents = [OllamaAgent(model_name) for _ in range(n_agents)]
        self.interaction_history = []
        self.consensus_history = []
        
    def run_interaction(self) -> Tuple[str, str, int]:
        """Ejecuta una interacción entre dos agentes aleatorios"""
        # Seleccionar dos agentes aleatorios
        agent1, agent2 = random.sample(self.agents, 2)
        
        # Cada agente elige una acción
        choice1 = agent1.choose_action(self.word_pool)
        choice2 = agent2.choose_action(self.word_pool)
        
        # Calcular payoff
        payoff = 100 if choice1 == choice2 else -50
        
        # Actualizar memoria de ambos agentes
        agent1.update_memory(choice1, choice2, payoff)
        agent2.update_memory(choice2, choice1, payoff)
        
        # Registrar interacción
        interaction = {
            'agent1_id': agent1.agent_id,
            'agent2_id': agent2.agent_id,
            'choice1': choice1,
            'choice2': choice2,
            'payoff': payoff,
            'round': len(self.interaction_history) + 1
        }
        self.interaction_history.append(interaction)
        
        return choice1, choice2, payoff
    
    def measure_consensus(self) -> Dict[str, float]:
        """Mide el nivel de consenso actual en la población"""
        if not self.interaction_history:
            return {word: 1.0/len(self.word_pool) for word in self.word_pool}
        
        # Contar las últimas elecciones de cada agente
        recent_choices = []
        for agent in self.agents:
            if agent.memory:
                # Tomar la elección más reciente como Player 1
                recent_choices.append(agent.memory[-1]['player1_choice'])
            else:
                # Si no tiene memoria, hacer una elección
                recent_choices.append(agent.choose_action(self.word_pool))
        
        # Calcular distribución
        total = len(recent_choices)
        consensus = {}
        for word in self.word_pool:
            consensus[word] = recent_choices.count(word) / total
        
        return consensus
    
    def run_simulation(self, n_rounds: int = 1000, verbose: bool = True) -> Dict:
        """Ejecuta la simulación completa"""
        print(f"Iniciando simulación con {self.n_agents} agentes, {n_rounds} rondas")
        print(f"Pool de palabras: {self.word_pool}")
        print("=" * 50)
        
        success_rate_history = []
        
        for round_num in range(n_rounds):
            choice1, choice2, payoff = self.run_interaction()
            
            # Medir consenso cada 50 rondas
            if round_num % 50 == 0:
                consensus = self.measure_consensus()
                self.consensus_history.append(consensus)
                
                # Calcular tasa de éxito reciente
                recent_interactions = self.interaction_history[-50:] if len(self.interaction_history) >= 50 else self.interaction_history
                successes = sum(1 for interaction in recent_interactions if interaction['payoff'] > 0)
                success_rate = successes / len(recent_interactions) if recent_interactions else 0
                success_rate_history.append(success_rate)
                
                if verbose:
                    print(f"Ronda {round_num:4d}: Consenso={consensus}, Éxito={success_rate:.2f}")
            
            # Pequeña pausa para evitar sobrecargar Ollama
            if round_num % 10 == 0:
                time.sleep(0.1)
        
        # Consenso final
        final_consensus = self.measure_consensus()
        self.consensus_history.append(final_consensus)
        
        return {
            'final_consensus': final_consensus,
            'consensus_history': self.consensus_history,
            'success_rate_history': success_rate_history,
            'total_interactions': len(self.interaction_history),
            'agents': self.agents
        }
    
    def analyze_individual_bias(self, n_trials: int = 100) -> Dict[str, float]:
        """Analiza el sesgo individual de los agentes (primera elección sin memoria)"""
        print("Analizando sesgo individual...")
        
        bias_results = {word: 0 for word in self.word_pool}
        
        for trial in range(n_trials):
            # Crear agente fresco sin memoria
            fresh_agent = OllamaAgent(self.agents[0].model_name)
            choice = fresh_agent.choose_action(self.word_pool)
            bias_results[choice] += 1
            
            if trial % 20 == 0:
                print(f"Progreso: {trial}/{n_trials}")
        
        # Convertir a proporciones
        for word in bias_results:
            bias_results[word] /= n_trials
        
        return bias_results
    
    def plot_results(self, results: Dict):
        """Genera gráficas de los resultados"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Evolución del consenso
        consensus_data = {word: [] for word in self.word_pool}
        for consensus in results['consensus_history']:
            for word in self.word_pool:
                consensus_data[word].append(consensus.get(word, 0))
        
        for word in self.word_pool:
            ax1.plot(consensus_data[word], label=f"Palabra '{word}'", marker='o')
        ax1.set_xlabel('Mediciones (cada 50 rondas)')
        ax1.set_ylabel('Proporción de Consenso')
        ax1.set_title('Evolución del Consenso')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Tasa de éxito
        ax2.plot(results['success_rate_history'], 'g-', marker='s')
        ax2.set_xlabel('Mediciones (cada 50 rondas)')
        ax2.set_ylabel('Tasa de Éxito')
        ax2.set_title('Evolución de la Coordinación')
        ax2.grid(True)
        
        # 3. Consenso final
        words = list(results['final_consensus'].keys())
        proportions = list(results['final_consensus'].values())
        ax3.bar(words, proportions, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(words)])
        ax3.set_xlabel('Palabra')
        ax3.set_ylabel('Proporción Final')
        ax3.set_title('Consenso Final')
        ax3.grid(True, axis='y')
        
        # 4. Distribución de puntuaciones
        scores = [agent.score for agent in self.agents]
        ax4.hist(scores, bins=10, alpha=0.7, color='purple')
        ax4.set_xlabel('Puntuación Final')
        ax4.set_ylabel('Número de Agentes')
        ax4.set_title('Distribución de Puntuaciones')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('naming_game_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Función principal para ejecutar experimentos"""
    
    print("🎮 JUEGO DE NOMBRES - IMPLEMENTACIÓN BASADA EN EL ARTÍCULO")
    print("=" * 60)
    
    # Configuración del experimento
    config = {
        'n_agents': 8,
        'word_pool': ['F', 'J'],  # Como en el artículo
        'n_rounds': 200,
        'model_name': 'llama3'
    }
    
    print(f"Configuración: {config}")
    print("\n🔧 Verificando conexión con Ollama...")
    
    # Verificar que Ollama esté corriendo
    try:
        test_agent = OllamaAgent(config['model_name'])
        test_response = test_agent._make_ollama_request("Hello, just testing connection.")
        print("✅ Conexión con Ollama exitosa!")
    except Exception as e:
        print(f"❌ Error conectando con Ollama: {e}")
        print("Asegúrate de que Ollama esté corriendo en localhost:11434")
        return
    
    # Crear y ejecutar simulación
    print("\n🚀 Iniciando simulación...")
    simulation = NamingGameSimulation(
        n_agents=config['n_agents'],
        word_pool=config['word_pool'],
        model_name=config['model_name']
    )
    
    # Ejecutar simulación principal
    results = simulation.run_simulation(n_rounds=config['n_rounds'])
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES")
    print("="*50)
    print(f"Consenso final: {results['final_consensus']}")
    print(f"Total de interacciones: {results['total_interactions']}")
    
    # Encontrar palabra dominante
    dominant_word = max(results['final_consensus'].items(), key=lambda x: x[1])
    print(f"Palabra dominante: '{dominant_word[0]}' ({dominant_word[1]:.2%})")
    
    # Calcular tasa de éxito final
    final_success_rate = results['success_rate_history'][-1] if results['success_rate_history'] else 0
    print(f"Tasa de éxito final: {final_success_rate:.2%}")
    
    # Analizar sesgo individual
    print("\n🔍 Analizando sesgo individual...")
    bias_results = simulation.analyze_individual_bias(n_trials=50)
    print(f"Sesgo individual (primera elección): {bias_results}")
    
    # Generar gráficas
    print("\n📈 Generando visualizaciones...")
    simulation.plot_results(results)
    
    # Guardar resultados
    results_summary = {
        'config': config,
        'final_consensus': results['final_consensus'],
        'individual_bias': bias_results,
        'final_success_rate': final_success_rate,
        'total_interactions': results['total_interactions']
    }
    
    with open('naming_game_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✅ Simulación completada!")
    print("📁 Resultados guardados en: naming_game_results.json")
    print("📊 Gráficas guardadas en: naming_game_results.png")

if __name__ == "__main__":
    main()