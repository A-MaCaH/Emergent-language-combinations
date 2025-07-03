#!/usr/bin/env python3
"""
Implementación del Juego de Nombres (Naming Game) usando Ollama y Llama3
Experimento ampliado para analizar el efecto del tamaño del word_pool en la convergencia
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
import pandas as pd
import seaborn as sns

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
            return random.choice(['F', 'J', 'A', 'B', 'C'])  # Fallback más robusto
    
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

class ConvergenceAnalyzer:
    """Analiza la convergencia del sistema"""
    
    @staticmethod
    def calculate_entropy(distribution: Dict[str, float]) -> float:
        """Calcula la entropía de Shannon de una distribución"""
        entropy = 0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    @staticmethod
    def detect_convergence(consensus_history: List[Dict[str, float]], 
                          threshold: float = 0.8, 
                          stability_rounds: int = 5) -> Tuple[bool, int]:
        """
        Detecta si el sistema ha convergido
        Returns: (has_converged, convergence_round)
        """
        if len(consensus_history) < stability_rounds:
            return False, -1
        
        for i in range(len(consensus_history) - stability_rounds + 1):
            # Verificar si hay una palabra dominante por stability_rounds consecutivos
            stable = True
            for j in range(stability_rounds):
                max_consensus = max(consensus_history[i + j].values())
                if max_consensus < threshold:
                    stable = False
                    break
            
            if stable:
                return True, i
        
        return False, -1
    
    @staticmethod
    def calculate_convergence_speed(consensus_history: List[Dict[str, float]], 
                                  threshold: float = 0.8) -> int:
        """Calcula la velocidad de convergencia (pasos hasta alcanzar threshold)"""
        for i, consensus in enumerate(consensus_history):
            max_consensus = max(consensus.values())
            if max_consensus >= threshold:
                return i
        return len(consensus_history)  # No convergió

class NamingGameSimulation:
    """Simulación del juego de nombres"""
    
    def __init__(self, n_agents: int = 10, word_pool: List[str] = None, 
                 model_name: str = "llama3"):
        self.n_agents = n_agents
        self.word_pool = word_pool or ['F', 'J']  # Pool por defecto del artículo
        self.agents = [OllamaAgent(model_name) for _ in range(n_agents)]
        self.interaction_history = []
        self.consensus_history = []
        self.analyzer = ConvergenceAnalyzer()
        
    def reset_simulation(self):
        """Reinicia la simulación con nuevos agentes"""
        self.agents = [OllamaAgent(self.agents[0].model_name) for _ in range(self.n_agents)]
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
    
    def run_simulation(self, n_rounds: int = 1000, verbose: bool = True, 
                      check_convergence: bool = True) -> Dict:
        """Ejecuta la simulación completa"""
        if verbose:
            print(f"Simulación con {self.n_agents} agentes, pool: {self.word_pool} ({len(self.word_pool)} opciones)")
        
        success_rate_history = []
        entropy_history = []
        converged = False
        convergence_round = -1
        
        for round_num in range(n_rounds):
            choice1, choice2, payoff = self.run_interaction()
            
            # Medir consenso cada 25 rondas (más frecuente para mejor detección)
            if round_num % 25 == 0:
                consensus = self.measure_consensus()
                self.consensus_history.append(consensus)
                
                # Calcular entropía
                entropy = self.analyzer.calculate_entropy(consensus)
                entropy_history.append(entropy)
                
                # Calcular tasa de éxito reciente
                recent_interactions = self.interaction_history[-25:] if len(self.interaction_history) >= 25 else self.interaction_history
                successes = sum(1 for interaction in recent_interactions if interaction['payoff'] > 0)
                success_rate = successes / len(recent_interactions) if recent_interactions else 0
                success_rate_history.append(success_rate)
                
                # Verificar convergencia
                if check_convergence and not converged:
                    converged, conv_round = self.analyzer.detect_convergence(
                        self.consensus_history, threshold=0.8, stability_rounds=3
                    )
                    if converged:
                        convergence_round = conv_round * 25  # Convertir a rondas reales
                        if verbose:
                            print(f"  🎯 ¡Convergencia detectada en ronda {convergence_round}!")
                
                if verbose and round_num % 100 == 0:
                    dominant_word = max(consensus.items(), key=lambda x: x[1])
                    print(f"  Ronda {round_num:4d}: Dominante='{dominant_word[0]}'({dominant_word[1]:.2f}), Éxito={success_rate:.2f}, Entropía={entropy:.2f}")
            
            # Pausa para evitar sobrecargar Ollama
            if round_num % 20 == 0:
                time.sleep(0.05)
            
            # Terminar temprano si ya convergió
            if converged and check_convergence and round_num > convergence_round + 100:
                if verbose:
                    print(f"  ✅ Terminando temprano tras convergencia estable")
                break
        
        # Consenso final
        final_consensus = self.measure_consensus()
        final_entropy = self.analyzer.calculate_entropy(final_consensus)
        
        return {
            'final_consensus': final_consensus,
            'consensus_history': self.consensus_history,
            'success_rate_history': success_rate_history,
            'entropy_history': entropy_history,
            'total_interactions': len(self.interaction_history),
            'converged': converged,
            'convergence_round': convergence_round,
            'convergence_speed': self.analyzer.calculate_convergence_speed(self.consensus_history),
            'final_entropy': final_entropy,
            'word_pool_size': len(self.word_pool)
        }

class MultiPoolExperiment:
    """Experimento con múltiples tamaños de word_pool"""
    
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.results = {}
        
        # Definir pools de diferentes tamaños
        self.word_pools = {
            2: ['A', 'B'],
            5: ['A', 'B', 'C', 'D', 'E'],
            10: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            50: [f'W{i:02d}' for i in range(50)],  # W00, W01, ..., W49
            100: [f'W{i:03d}' for i in range(100)]  # W000, W001, ..., W099
        }
    
    def run_experiment(self, n_agents: int = 8, n_rounds: int = 800, 
                      n_trials: int = 3, verbose: bool = True) -> Dict:
        """Ejecuta el experimento completo"""
        
        print("🧪 EXPERIMENTO MULTI-POOL - ANÁLISIS DE CONVERGENCIA")
        print("=" * 60)
        print(f"Configuración: {n_agents} agentes, {n_rounds} rondas máx, {n_trials} ensayos por pool")
        print("Pool sizes:", list(self.word_pools.keys()))
        print()
        
        all_results = {}
        
        for pool_size in self.word_pools.keys():
            print(f"🔬 Testando pool de tamaño {pool_size}...")
            
            pool_results = []
            word_pool = self.word_pools[pool_size]
            
            for trial in range(n_trials):
                print(f"  Ensayo {trial + 1}/{n_trials}")
                
                # Crear nueva simulación para cada ensayo
                simulation = NamingGameSimulation(
                    n_agents=n_agents,
                    word_pool=word_pool,
                    model_name=self.model_name
                )
                
                # Ejecutar simulación
                result = simulation.run_simulation(
                    n_rounds=n_rounds, 
                    verbose=False,
                    check_convergence=True
                )
                
                pool_results.append(result)
                
                # Mostrar progreso
                conv_status = "✅" if result['converged'] else "❌"
                conv_round = result['convergence_round'] if result['converged'] else "N/A"
                print(f"    {conv_status} Convergencia: {conv_round}, Entropía final: {result['final_entropy']:.2f}")
            
            # Agregar resultados agregados
            all_results[pool_size] = {
                'trials': pool_results,
                'convergence_rate': sum(1 for r in pool_results if r['converged']) / len(pool_results),
                'avg_convergence_speed': statistics.mean([r['convergence_speed'] for r in pool_results]),
                'avg_final_entropy': statistics.mean([r['final_entropy'] for r in pool_results]),
                'avg_final_success_rate': statistics.mean([r['success_rate_history'][-1] for r in pool_results if r['success_rate_history']])
            }
            
            print(f"  📊 Resumen pool {pool_size}:")
            print(f"    Tasa de convergencia: {all_results[pool_size]['convergence_rate']:.1%}")
            print(f"    Velocidad promedio: {all_results[pool_size]['avg_convergence_speed']:.1f} pasos")
            print(f"    Entropía final promedio: {all_results[pool_size]['avg_final_entropy']:.2f}")
            print()
        
        self.results = all_results
        return all_results
    
    def analyze_results(self) -> Dict:
        """Analiza los resultados del experimento"""
        if not self.results:
            print("❌ No hay resultados para analizar. Ejecuta run_experiment() primero.")
            return {}
        
        print("📈 ANÁLISIS DE RESULTADOS")
        print("=" * 40)
        
        analysis = {
            'pool_sizes': [],
            'convergence_rates': [],
            'avg_speeds': [],
            'avg_entropies': [],
            'avg_success_rates': []
        }
        
        for pool_size, results in self.results.items():
            analysis['pool_sizes'].append(pool_size)
            analysis['convergence_rates'].append(results['convergence_rate'])
            analysis['avg_speeds'].append(results['avg_convergence_speed'])
            analysis['avg_entropies'].append(results['avg_final_entropy'])
            analysis['avg_success_rates'].append(results['avg_final_success_rate'])
            
            print(f"Pool {pool_size:3d}: Conv={results['convergence_rate']:.1%}, "
                  f"Velocidad={results['avg_convergence_speed']:5.1f}, "
                  f"Entropía={results['avg_final_entropy']:.2f}")
        
        # Encontrar tendencias
        print("\n🔍 TENDENCIAS OBSERVADAS:")
        
        # Correlación entre tamaño del pool y convergencia
        correlation_speed = np.corrcoef(analysis['pool_sizes'], analysis['avg_speeds'])[0, 1]
        correlation_conv_rate = np.corrcoef(analysis['pool_sizes'], analysis['convergence_rates'])[0, 1]
        
        print(f"Correlación tamaño-velocidad: {correlation_speed:.3f}")
        print(f"Correlación tamaño-tasa_convergencia: {correlation_conv_rate:.3f}")
        
        return analysis
    
    def plot_comprehensive_results(self, analysis: Dict):
        """Genera gráficas comprehensivas de los resultados"""
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Tasa de convergencia vs tamaño del pool
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(analysis['pool_sizes'], analysis['convergence_rates'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Tamaño del Word Pool')
        plt.ylabel('Tasa de Convergencia')
        plt.title('Tasa de Convergencia vs Tamaño del Pool')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # 2. Velocidad de convergencia vs tamaño del pool
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(analysis['pool_sizes'], analysis['avg_speeds'], 's-', color='red', linewidth=2, markersize=8)
        plt.xlabel('Tamaño del Word Pool')
        plt.ylabel('Pasos hasta Convergencia')
        plt.title('Velocidad de Convergencia vs Tamaño del Pool')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala logarítmica para mejor visualización
        
        # 3. Entropía final vs tamaño del pool
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(analysis['pool_sizes'], analysis['avg_entropies'], '^-', color='green', linewidth=2, markersize=8)
        plt.xlabel('Tamaño del Word Pool')
        plt.ylabel('Entropía Final Promedio')
        plt.title('Entropía Final vs Tamaño del Pool')
        plt.grid(True, alpha=0.3)
        
        # 4. Distribución de velocidades de convergencia (boxplot)
        ax4 = plt.subplot(2, 3, 4)
        speeds_data = []
        labels = []
        for pool_size in analysis['pool_sizes']:
            if pool_size in self.results:
                speeds = [r['convergence_speed'] for r in self.results[pool_size]['trials']]
                speeds_data.append(speeds)
                labels.append(f'Pool {pool_size}')
        
        plt.boxplot(speeds_data, labels=labels)
        plt.ylabel('Pasos hasta Convergencia')
        plt.title('Distribución de Velocidades de Convergencia')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Heatmap de correlaciones
        ax5 = plt.subplot(2, 3, 5)
        correlation_data = np.array([
            analysis['pool_sizes'],
            analysis['convergence_rates'],
            analysis['avg_speeds'],
            analysis['avg_entropies']
        ]).T
        
        corr_matrix = np.corrcoef(correlation_data.T)
        labels_corr = ['Pool Size', 'Conv. Rate', 'Speed', 'Entropy']
        
        im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.xticks(range(len(labels_corr)), labels_corr, rotation=45)
        plt.yticks(range(len(labels_corr)), labels_corr)
        plt.title('Matriz de Correlaciones')
        plt.colorbar(im)
        
        # Agregar valores de correlación
        for i in range(len(labels_corr)):
            for j in range(len(labels_corr)):
                plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        # 6. Evolución temporal promedio
        ax6 = plt.subplot(2, 3, 6)
        for pool_size in [2, 5, 10, 50]:  # Mostrar solo algunos para claridad
            if pool_size in self.results:
                # Promediar las historias de entropía de todos los ensayos
                all_entropy_histories = []
                for trial in self.results[pool_size]['trials']:
                    if trial['entropy_history']:
                        all_entropy_histories.append(trial['entropy_history'])
                
                if all_entropy_histories:
                    # Encontrar la longitud mínima para poder promediar
                    min_length = min(len(hist) for hist in all_entropy_histories)
                    avg_entropy = np.mean([hist[:min_length] for hist in all_entropy_histories], axis=0)
                    x_axis = np.arange(len(avg_entropy)) * 25  # Cada medición es cada 25 rondas
                    plt.plot(x_axis, avg_entropy, label=f'Pool {pool_size}', linewidth=2)
        
        plt.xlabel('Rondas')
        plt.ylabel('Entropía Promedio')
        plt.title('Evolución Temporal de la Entropía')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_pool_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Función principal para ejecutar el experimento ampliado"""
    
    print("🎮 EXPERIMENTO AMPLIADO - NAMING GAME")
    print("Análisis del efecto del tamaño del word_pool en la convergencia")
    print("=" * 70)
    
    # Verificar conexión con Ollama
    print("🔧 Verificando conexión con Ollama...")
    try:
        test_agent = OllamaAgent('llama3')
        test_response = test_agent._make_ollama_request("Test")
        print("✅ Conexión exitosa!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de que Ollama esté corriendo en localhost:11434")
        return
    
    # Configuración del experimento
    config = {
        'n_agents': 6,  # Reducido para acelerar el experimento
        'n_rounds': 600,  # Suficiente para detectar convergencia
        'n_trials': 3,  # Múltiples ensayos para robustez estadística
        'model_name': 'llama3'
    }
    
    print(f"\n📋 Configuración del experimento:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Crear y ejecutar experimento
    experiment = MultiPoolExperiment(config['model_name'])
    
    print(f"\n🚀 Iniciando experimento...")
    start_time = time.time()
    
    results = experiment.run_experiment(
        n_agents=config['n_agents'],
        n_rounds=config['n_rounds'],
        n_trials=config['n_trials']
    )
    
    end_time = time.time()
    print(f"⏱️  Tiempo total: {end_time - start_time:.1f} segundos")
    
    # Analizar resultados
    analysis = experiment.analyze_results()
    
    # Generar visualizaciones
    print("\n📊 Generando visualizaciones...")
    experiment.plot_comprehensive_results(analysis)
    
    # Guardar resultados
    output_data = {
        'config': config,
        'results': results,
        'analysis': analysis,
        'execution_time': end_time - start_time
    }
    
    with open('multi_pool_experiment_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print("\n✅ Experimento completado!")
    print("📁 Resultados guardados en: multi_pool_experiment_results.json")
    print("📊 Visualizaciones guardadas en: multi_pool_experiment_results.png")
    
    # Mostrar conclusiones clave
    print("\n" + "="*50)
    print("🎯 CONCLUSIONES CLAVE")
    print("="*50)
    
    if analysis:
        fastest_pool = min(enumerate(analysis['avg_speeds']), key=lambda x: x[1])
        slowest_pool = max(enumerate(analysis['avg_speeds']), key=lambda x: x[1])
        
        print(f"Pool más rápido en converger: {analysis['pool_sizes'][fastest_pool[0]]} ({fastest_pool[1]:.1f} pasos)")
        print(f"Pool más lento en converger: {analysis['pool_sizes'][slowest_pool[0]]} ({slowest_pool[1]:.1f} pasos)")
        
        # Calcular factor de escalamiento
        speed_ratio = slowest_pool[1] / fastest_pool[1]
        size_ratio = analysis['pool_sizes'][slowest_pool[0]] / analysis['pool_sizes'][fastest_pool[0]]
        
        print(f"Factor de escalamiento en velocidad: {speed_ratio:.1f}x")
        print(f"Factor de escalamiento en tamaño: {size_ratio:.1f}x")
        
        # Tasa de éxito en convergencia
        avg_conv_rate = statistics.mean(analysis['convergence_rates'])
        print(f"Tasa promedio de convergencia: {avg_conv_rate:.1%}")

if __name__ == "__main__":
    main()
