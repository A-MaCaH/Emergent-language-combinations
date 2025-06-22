#!/usr/bin/env python3
"""
🎮 JUEGO DE NOMBRES (NAMING GAME) - VERSIÓN EDUCATIVA
===================================================

¿QUÉ ESTÁ PASANDO EN ESTE EXPERIMENTO?

Imagina que tienes un grupo de robots (agentes LLM) que ven el MISMO OBJETO pero 
no saben cómo llamarlo. Cada robot debe elegir un nombre de una lista de opciones.

🎯 OBJETIVO: Todos los robots deben ponerse de acuerdo en usar el MISMO nombre.

📋 REGLAS:
- Si 2 robots eligen el MISMO nombre → ambos ganan +100 puntos ✅
- Si 2 robots eligen nombres DIFERENTES → ambos pierden -50 puntos ❌

🧠 LO FASCINANTE: Sin comunicación directa, solo viendo los resultados de 
   interacciones pasadas, los robots aprenden a coordinarse y formar CONSENSOS.

Basado en el artículo científico sobre emergencia de lenguajes en LLMs.
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
    """🤖 Agente robot que usa Ollama/Llama3 para tomar decisiones"""
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.7, 
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.memory = []  # 🧠 Memoria de interacciones pasadas
        self.memory_size = 5  # Recuerda últimas 5 interacciones
        self.score = 0
        self.agent_id = f"Robot_{random.randint(100, 999)}"
        
    def _make_ollama_request(self, prompt: str) -> str:
        """📡 Envía petición al modelo Llama3 via Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_k": 40,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"❌ Error conectando con Ollama: {e}")
            return random.choice(['A', 'B'])  # Fallback aleatorio
    
    def _build_system_prompt(self, word_pool: List[str], target_object: str) -> str:
        """🏗️ Construye el prompt que describe la situación al robot"""
        
        # Construir historial de memoria
        memory_str = ""
        if self.memory:
            memory_str = "\n🕐 HISTORIAL DE TUS INTERACCIONES PASADAS:\n"
            for i, interaction in enumerate(self.memory[-self.memory_size:], 1):
                result_emoji = "✅" if interaction['payoff'] > 0 else "❌"
                memory_str += f"   Ronda {len(self.memory) - self.memory_size + i}: "
                memory_str += f"Tú elegiste '{interaction['my_choice']}', "
                memory_str += f"otro robot eligió '{interaction['other_choice']}' "
                memory_str += f"{result_emoji} ({interaction['payoff']:+d} puntos)\n"
        else:
            memory_str = "\n🆕 PRIMERA INTERACCIÓN: No tienes historial previo.\n"
        
        system_prompt = f"""🎯 SITUACIÓN: Eres un robot inteligente participando en un experimento de coordinación.

🔍 LO QUE VES: {target_object}

📝 TU MISIÓN: Debes elegir un nombre para este objeto de la siguiente lista:
   Opciones disponibles: {word_pool}

⚡ REGLAS DEL JUEGO:
   • Si TÚ y OTRO ROBOT eligen el MISMO nombre → ambos ganan +100 puntos ✅
   • Si eligen nombres DIFERENTES → ambos pierden -50 puntos ❌
   • Tu objetivo: MAXIMIZAR tus puntos coordinándote con otros robots

{memory_str}
📊 TU PUNTUACIÓN ACTUAL: {self.score} puntos
🎲 RONDA ACTUAL: {len(self.memory) + 1}

🤔 INSTRUCCIONES:
1. Analiza tu historial (si lo tienes)
2. Piensa qué nombre es más probable que otros robots elijan
3. Elige estratégicamente para maximizar coordinación

Responde SOLO en este formato JSON:
{{"name": "<NOMBRE_ELEGIDO>", "reasoning": "<TU_RAZONAMIENTO>"}}

IMPORTANTE: Debes elegir exactamente uno de: {word_pool}"""

        return system_prompt
    
    def choose_name(self, word_pool: List[str], target_object: str) -> str:
        """🎯 El robot elige un nombre para el objeto"""
        
        # En la primera ronda, elección aleatoria
        if not self.memory:
            choice = random.choice(word_pool)
            print(f"   🤖 {self.agent_id}: Primera vez, elijo '{choice}' aleatoriamente")
            return choice
        
        # Construir prompt y obtener respuesta del LLM
        system_prompt = self._build_system_prompt(word_pool, target_object)
        
        response = self._make_ollama_request(system_prompt)
        
        # Extraer decisión del JSON
        try:
            # Buscar JSON en la respuesta
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                choice = parsed.get('name', '').strip().upper()
                reasoning = parsed.get('reasoning', 'Sin razonamiento')
                
                if choice in word_pool:
                    print(f"   🤖 {self.agent_id}: Elijo '{choice}' - {reasoning[:50]}...")
                    return choice
        except Exception as e:
            print(f"   ⚠️ {self.agent_id}: Error parseando respuesta")
        
        # Fallback: buscar palabras del pool en la respuesta
        for word in word_pool:
            if word.upper() in response.upper():
                print(f"   🤖 {self.agent_id}: Detecto '{word}' en la respuesta")
                return word
        
        # Último fallback: elección aleatoria
        choice = random.choice(word_pool)
        print(f"   🤖 {self.agent_id}: Fallback aleatorio → '{choice}'")
        return choice
    
    def update_memory(self, my_choice: str, other_choice: str, payoff: int):
        """🧠 Actualiza la memoria del robot con nueva experiencia"""
        self.memory.append({
            'my_choice': my_choice,
            'other_choice': other_choice,
            'payoff': payoff,
            'round': len(self.memory) + 1
        })
        self.score += payoff
        
        # Mantener solo últimas interacciones en memoria
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

class NamingGameExperiment:
    """🔬 Experimento del Juego de Nombres"""
    
    def __init__(self, n_agents: int = 6, word_pool: List[str] = None, 
                 target_object: str = None, model_name: str = "llama3"):
        self.n_agents = n_agents
        self.word_pool = word_pool or ['MANZANA', 'FRUTA']
        self.target_object = target_object or "una fruta roja y redonda"
        self.agents = [OllamaAgent(model_name) for _ in range(n_agents)]
        self.interaction_history = []
        self.consensus_history = []
        
        # Asignar IDs más descriptivos
        for i, agent in enumerate(self.agents):
            agent.agent_id = f"Robot_{i+1:02d}"
    
    def display_experiment_info(self):
        """📋 Muestra información del experimento"""
        print("\n" + "="*80)
        print("🧪 CONFIGURACIÓN DEL EXPERIMENTO")
        print("="*80)
        print(f"👥 Número de robots participantes: {self.n_agents}")
        print(f"🎯 Objeto que todos ven: {self.target_object}")
        print(f"📝 Nombres posibles para elegir: {self.word_pool}")
        print(f"🧠 Modelo de IA usado: Llama3 via Ollama")
        print(f"💭 Memoria de cada robot: {self.agents[0].memory_size} interacciones")
        
        print(f"\n🎮 MECÁNICA DEL JUEGO:")
        print(f"   • Cada ronda: se eligen 2 robots al azar")
        print(f"   • Ambos ven '{self.target_object}'")
        print(f"   • Cada uno elige independientemente un nombre: {self.word_pool}")
        print(f"   • Si eligen igual → +100 puntos cada uno ✅")
        print(f"   • Si eligen diferente → -50 puntos cada uno ❌")
        print(f"   • Los robots aprenden del historial de interacciones")
        
        print(f"\n🔬 LO QUE ESTAMOS ESTUDIANDO:")
        print(f"   1. ¿Emergerá un CONSENSO sobre cómo llamar al objeto?")
        print(f"   2. ¿Qué nombre 'ganará' y por qué?")
        print(f"   3. ¿Cuánto tiempo toma llegar al consenso?")
        print(f"   4. ¿Los robots muestran sesgos hacia ciertos nombres?")
        print("="*80)
    
    def run_single_interaction(self, round_num: int) -> Tuple[str, str, int]:
        """🎲 Ejecuta una sola interacción entre dos robots"""
        
        # Seleccionar dos robots aleatorios
        robot1, robot2 = random.sample(self.agents, 2)
        
        print(f"\n🎭 RONDA {round_num}")
        print(f"   Participantes: {robot1.agent_id} vs {robot2.agent_id}")
        print(f"   Ambos ven: '{self.target_object}'")
        print(f"   Deben elegir entre: {self.word_pool}")
        
        # Cada robot elige un nombre
        choice1 = robot1.choose_name(self.word_pool, self.target_object)
        choice2 = robot2.choose_name(self.word_pool, self.target_object)
        
        # Calcular resultado
        payoff = 100 if choice1 == choice2 else -50
        result_emoji = "✅" if payoff > 0 else "❌"
        
        print(f"   📊 RESULTADO: '{choice1}' vs '{choice2}' → {result_emoji} {payoff:+d} puntos c/u")
        
        # Actualizar memorias
        robot1.update_memory(choice1, choice2, payoff)
        robot2.update_memory(choice2, choice1, payoff)
        
        # Registrar interacción
        interaction = {
            'round': round_num,
            'robot1_id': robot1.agent_id,
            'robot2_id': robot2.agent_id,
            'choice1': choice1,
            'choice2': choice2,
            'payoff': payoff,
            'success': payoff > 0
        }
        self.interaction_history.append(interaction)
        
        return choice1, choice2, payoff
    
    def measure_current_consensus(self) -> Dict[str, float]:
        """📊 Mide el consenso actual en la población"""
        
        print(f"\n🔍 MIDIENDO CONSENSO ACTUAL...")
        
        # Obtener la preferencia actual de cada robot
        current_preferences = []
        for agent in self.agents:
            if agent.memory:
                # Usar la última elección como indicador de preferencia
                last_choice = agent.memory[-1]['my_choice']
                current_preferences.append(last_choice)
                print(f"   {agent.agent_id}: prefiere '{last_choice}' (basado en experiencia)")
            else:
                # Robot sin experiencia - simular elección
                test_choice = agent.choose_name(self.word_pool, self.target_object)
                current_preferences.append(test_choice)
                print(f"   {agent.agent_id}: sin experiencia, elige '{test_choice}'")
        
        # Calcular distribución de consenso
        total = len(current_preferences)
        consensus = {}
        for word in self.word_pool:
            count = current_preferences.count(word)
            percentage = (count / total) * 100
            consensus[word] = count / total
            print(f"   📈 '{word}': {count}/{total} robots ({percentage:.1f}%)")
        
        return consensus, current_preferences
    
    def run_full_experiment(self, n_rounds: int = 50) -> Dict:
        """🚀 Ejecuta el experimento completo"""
        
        self.display_experiment_info()
        
        input("\n⏸️  Presiona ENTER para comenzar el experimento...")
        
        print(f"\n🎬 INICIANDO EXPERIMENTO - {n_rounds} RONDAS")
        print("="*80)
        
        success_count = 0
        consensus_measurements = []
        
        for round_num in range(1, n_rounds + 1):
            choice1, choice2, payoff = self.run_single_interaction(round_num)
            
            if payoff > 0:
                success_count += 1
            
            # Medir consenso cada 10 rondas
            if round_num % 10 == 0 or round_num == n_rounds:
                print(f"\n" + "="*50)
                print(f"📊 MEDICIÓN DE CONSENSO - RONDA {round_num}")
                print("="*50)
                
                consensus_data, preferences = self.measure_current_consensus()
                consensus_measurements.append({
                    'round': round_num,
                    'consensus': consensus_data,
                    'preferences': preferences
                })
                
                success_rate = success_count / round_num
                print(f"📈 Tasa de éxito acumulada: {success_rate:.1%} ({success_count}/{round_num})")
                
                # Mostrar robot con mejor puntuación
                best_robot = max(self.agents, key=lambda x: x.score)
                print(f"🏆 Robot líder: {best_robot.agent_id} con {best_robot.score} puntos")
                
                if round_num < n_rounds:
                    input(f"\n⏸️  Presiona ENTER para continuar...")
            
            # Pausa pequeña para no sobrecargar
            time.sleep(0.1)
        
        # Análisis final
        return self.analyze_final_results(consensus_measurements, success_count, n_rounds)
    
    def analyze_final_results(self, consensus_measurements: List, success_count: int, n_rounds: int) -> Dict:
        """📊 Analiza y presenta los resultados finales"""
        
        print(f"\n" + "="*80)
        print("🎉 EXPERIMENTO COMPLETADO - ANÁLISIS FINAL")
        print("="*80)
        
        final_consensus = consensus_measurements[-1]['consensus'] if consensus_measurements else {}
        final_preferences = consensus_measurements[-1]['preferences'] if consensus_measurements else []
        
        # Encontrar nombre ganador
        if final_consensus:
            winner_name = max(final_consensus.items(), key=lambda x: x[1])
            winner_percentage = winner_name[1] * 100
            
            print(f"🏆 NOMBRE GANADOR: '{winner_name[0]}' ({winner_percentage:.1f}% de consenso)")
            
            if winner_percentage >= 80:
                print("   ✨ ¡CONSENSO FUERTE ESTABLECIDO!")
            elif winner_percentage >= 60:
                print("   📈 Consenso moderado")
            else:
                print("   🤔 Consenso débil - población dividida")
        
        # Estadísticas generales
        success_rate = success_count / n_rounds
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        print(f"   • Interacciones exitosas: {success_count}/{n_rounds} ({success_rate:.1%})")
        print(f"   • Interacciones fallidas: {n_rounds - success_count}/{n_rounds} ({1-success_rate:.1%})")
        
        # Puntuaciones finales
        print(f"\n🏅 PUNTUACIONES FINALES:")
        sorted_agents = sorted(self.agents, key=lambda x: x.score, reverse=True)
        for i, agent in enumerate(sorted_agents):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"   {medal} {agent.agent_id}: {agent.score:+4d} puntos")
        
        # Evolución del consenso
        if len(consensus_measurements) > 1:
            print(f"\n📈 EVOLUCIÓN DEL CONSENSO:")
            for measurement in consensus_measurements:
                round_num = measurement['round']
                consensus = measurement['consensus']
                dominant = max(consensus.items(), key=lambda x: x[1])
                print(f"   Ronda {round_num:2d}: '{dominant[0]}' dominante ({dominant[1]:.1%})")
        
        # Conclusiones
        print(f"\n🔬 CONCLUSIONES DEL EXPERIMENTO:")
        if success_rate > 0.8:
            print("   ✅ Los robots aprendieron a coordinarse efectivamente")
        elif success_rate > 0.6:
            print("   📈 Los robots mostraron capacidad de coordinación moderada")
        else:
            print("   🤔 La coordinación fue limitada - puede necesitar más rondas")
        
        if final_consensus and max(final_consensus.values()) > 0.7:
            print("   ✅ Se estableció un consenso claro sobre el nombre del objeto")
        else:
            print("   🤔 No se logró consenso fuerte - población dividida")
        
        return {
            'final_consensus': final_consensus,
            'success_rate': success_rate,
            'winner_name': winner_name[0] if final_consensus else None,
            'consensus_measurements': consensus_measurements,
            'agent_scores': {agent.agent_id: agent.score for agent in self.agents}
        }

def main():
    """🎮 Función principal - Configurar y ejecutar experimento"""
    
    print("🎮" + "="*79)
    print("   JUEGO DE NOMBRES - EXPERIMENTO DE CONSENSO CON IA")
    print("   Basado en investigación sobre emergencia de lenguajes en LLMs")
    print("="*80)
    
    print("\n🎯 ¿QUÉ VAMOS A ESTUDIAR?")
    print("   Los robots ven un OBJETO y deben ponerse de acuerdo en su NOMBRE")
    print("   Sin comunicación directa, solo aprendiendo de interacciones pasadas")
    
    # Verificar Ollama
    print(f"\n🔧 Verificando conexión con Ollama...")
    try:
        test_agent = OllamaAgent()
        test_response = test_agent._make_ollama_request("Test connection")
        print("   ✅ Ollama conectado correctamente!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   💡 Asegúrate de que Ollama esté corriendo: 'ollama serve'")
        print("   💡 Y que tengas Llama3 instalado: 'ollama pull llama3'")
        return
    
    # Configurar experimento
    print(f"\n⚙️ CONFIGURANDO EXPERIMENTO...")
    
    experiments = [
        {
            'name': 'Fruta Roja',
            'object': 'una fruta roja y redonda que cabe en tu mano',
            'words': ['MANZANA', 'FRUTA'],
            'agents': 6
        },
        {
            'name': 'Animal Pequeño', 
            'object': 'un pequeño animal peludo de cuatro patas que maúlla',
            'words': ['GATO', 'FELINO', 'MASCOTA'],
            'agents': 6
        },
        {
            'name': 'Experimento Abstracto',
            'object': 'un símbolo que representa coordinación',
            'words': ['ALPHA', 'BETA'],
            'agents': 8
        }
    ]
    
    print("\nElige un experimento:")
    for i, exp in enumerate(experiments):
        print(f"   {i+1}. {exp['name']}: '{exp['object']}'")
        print(f"      Opciones: {exp['words']} | Robots: {exp['agents']}")
    
    try:
        choice = int(input("\nElige experimento (1-3): ")) - 1
        if 0 <= choice < len(experiments):
            config = experiments[choice]
        else:
            print("Opción inválida, usando experimento por defecto...")
            config = experiments[0]
    except:
        print("Entrada inválida, usando experimento por defecto...")
        config = experiments[0]
    
    # Crear y ejecutar experimento
    experiment = NamingGameExperiment(
        n_agents=config['agents'],
        word_pool=config['words'],
        target_object=config['object']
    )
    
    results = experiment.run_full_experiment(n_rounds=50)
    
    # Guardar resultados
    print(f"\n💾 Guardando resultados...")
    with open('naming_game_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Resultados guardados en: naming_game_results.json")
    print(f"\n🎉 ¡Experimento completado exitosamente!")
    print(f"   ¿Observaste cómo emergió el consenso? 🤔")

if __name__ == "__main__":
    main()