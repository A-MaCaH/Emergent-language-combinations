#!/usr/bin/env python3
"""
🎮 JUEGO DE NOMBRES AVANZADO - Invención de Palabras
Los agentes LLM deben inventar palabras completamente nuevas para describir un objeto específico
Basado en el artículo sobre emergencia de lenguajes en LLMs
"""

import json
import random
import requests
import time
from typing import List, Dict, Tuple, Optional, Set
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import statistics
import re

class InventiveOllamaAgent:
    """Agente que inventa palabras nuevas para objetos"""
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.8, 
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.memory = []  # Historial de interacciones
        self.memory_size = 5  # Tamaño de memoria
        self.score = 0
        self.agent_id = f"agent_{random.randint(1000, 9999)}"
        self.invented_words = set()  # Palabras que este agente ha inventado
        
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
                        "top_k": 40,
                        "top_p": 0.9,
                    }
                },
                timeout=45
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"❌ Error en petición a Ollama: {e}")
            return f"FALLBACK{random.randint(100,999)}"
    
    def _build_system_prompt(self, target_object: str, global_vocabulary: Set[str]) -> str:
        """Construye el prompt para invención de palabras"""
        
        # Construir historial
        memory_str = ""
        if self.memory:
            memory_str = "\n🔍 HISTORIAL DE INTERACCIONES ANTERIORES:\n"
            for i, interaction in enumerate(self.memory[-self.memory_size:], 1):
                success_emoji = "✅" if interaction['payoff'] > 0 else "❌"
                memory_str += f"Ronda {len(self.memory) - self.memory_size + i}: "
                memory_str += f"Yo dije '{interaction['my_word']}', "
                memory_str += f"Mi compañero dijo '{interaction['partner_word']}' "
                memory_str += f"{success_emoji} (Puntos: {interaction['payoff']})\n"
        
        # Mostrar vocabulario conocido
        vocab_str = ""
        if global_vocabulary:
            vocab_str = f"\n📚 PALABRAS YA USADAS POR TODOS: {', '.join(sorted(global_vocabulary))}\n"
        
        system_prompt = f"""🎯 DESAFÍO DE INVENCIÓN LINGÜÍSTICA

🌟 OBJETO MISTERIOSO: {target_object}

📋 REGLAS DEL JUEGO:
• Tú y tu compañero están viendo el mismo objeto: "{target_object}"
• Ambos deben INVENTAR una palabra completamente nueva para describirlo
• OBJETIVO: Coordinar para usar la MISMA palabra inventada

💰 SISTEMA DE PUNTOS:
• Si ambos usan la MISMA palabra inventada: +100 puntos cada uno ✅
• Si usan palabras DIFERENTES: -50 puntos cada uno ❌

🧠 ESTRATEGIA:
• Puedes inventar una palabra completamente nueva
• O puedes usar una palabra que ya funcionó antes
• Analiza el historial para ver qué estrategias han funcionado
• Recuerda: tu compañero también está tratando de coordinar contigo

{memory_str}{vocab_str}

🎯 PUNTUACIÓN ACTUAL: {self.score} puntos
🔢 RONDA ACTUAL: {len(self.memory) + 1}

INSTRUCCIONES:
1. Analiza el objeto: {target_object}
2. Considera tu historial de éxitos y fracasos
3. Inventa UNA palabra nueva o usa una que haya funcionado
4. La palabra debe ser creativa pero no demasiado larga (máximo 3 caracteres)
5. Responde SOLO con tu palabra inventada, sin explicaciones

¿Qué palabra inventas para "{target_object}"?"""

        return system_prompt
    
    def invent_word(self, target_object: str, global_vocabulary: Set[str]) -> str:
        """Inventa una palabra para el objeto objetivo"""
        
        if not self.memory:
            # Primera ronda: invención pura
            prompt = f"""Inventa una palabra completamente nueva y creativa para describir: {target_object}

La palabra debe ser:
- Completamente inventada (no una palabra real)
- Máximo 3 caracteres
- Fácil de recordar
- Creativa pero no demasiado complicada

Responde SOLO con la palabra inventada, nada más."""
            
            response = self._make_ollama_request(prompt)
            word = self._extract_word(response)
            self.invented_words.add(word)
            return word
        
        # Rondas posteriores: usar estrategia basada en historial
        system_prompt = self._build_system_prompt(target_object, global_vocabulary)
        response = self._make_ollama_request(system_prompt)
        word = self._extract_word(response)
        self.invented_words.add(word)
        return word
    
    def _extract_word(self, response: str) -> str:
        """Extrae la palabra inventada de la respuesta"""
        # Limpiar respuesta
        response = response.strip()
        
        # Buscar patrones comunes
        patterns = [
            r'"([^"]{1,10})"',  # Entre comillas
            r'palabra[:\s]+([A-Za-z0-9]{1,10})',  # Después de "palabra:"
            r'^([A-Za-z0-9]{1,10})\s*$',  # Línea simple
            r'invento[:\s]+([A-Za-z0-9]{1,10})',  # Después de "invento:"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                word = match.group(1).upper()
                if len(word) <= 10 and word.isalnum():
                    return word
        
        # Buscar primera palabra válida
        words = re.findall(r'\b([A-Za-z0-9]{2,10})\b', response)
        for word in words:
            if word.upper() not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'GET', 'MAY', 'SAY', 'SHE', 'USE', 'DAY', 'RUN', 'GOT', 'LET', 'PUT', 'TRY']:
                return word.upper()
        
        # Fallback: generar palabra aleatoria
        consonants = 'BCDFGHJKLMNPQRSTVWXYZ'
        vowels = 'AEIOU'
        word = random.choice(consonants) + random.choice(vowels) + random.choice(consonants)
        return word + str(random.randint(10, 99))
    
    def update_memory(self, my_word: str, partner_word: str, payoff: int):
        """Actualiza la memoria del agente"""
        self.memory.append({
            'my_word': my_word,
            'partner_word': partner_word,
            'payoff': payoff,
            'round': len(self.memory) + 1
        })
        self.score += payoff
        
        # Mantener memoria limitada
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

class CreativaNamingGameSimulation:
    """Simulación avanzada del juego de nombres con invención de palabras"""
    
    def __init__(self, n_agents: int = 6, target_object: str = "una criatura extraña con tentáculos brillantes", 
                 model_name: str = "llama3"):
        self.n_agents = n_agents
        self.target_object = target_object
        self.agents = [InventiveOllamaAgent(model_name) for _ in range(n_agents)]
        self.global_vocabulary = set()  # Todas las palabras inventadas
        self.interaction_history = []
        self.consensus_history = []
        self.word_frequency = defaultdict(int)
        
        # Imprimir información inicial
        print(f"\n🎮 INICIANDO EXPERIMENTO DE INVENCIÓN LINGÜÍSTICA")
        print(f"🎯 OBJETO MISTERIOSO: {self.target_object}")
        print(f"👥 NÚMERO DE AGENTES: {self.n_agents}")
        print(f"🤖 MODELO: {model_name}")
        print("=" * 80)
    
    def run_interaction(self, round_num: int) -> Tuple[str, str, int]:
        """Ejecuta una interacción entre dos agentes aleatorios"""
        # Seleccionar dos agentes aleatorios
        agent1, agent2 = random.sample(self.agents, 2)
        
        print(f"\n🔄 RONDA {round_num}")
        print(f"🤝 Interacción entre {agent1.agent_id} y {agent2.agent_id}")
        print(f"👁️  Ambos ven: {self.target_object}")
        
        # Cada agente inventa una palabra
        print("🧠 Los agentes están pensando...")
        word1 = agent1.invent_word(self.target_object, self.global_vocabulary)
        word2 = agent2.invent_word(self.target_object, self.global_vocabulary)
        
        # Mostrar palabras inventadas
        print(f"💭 {agent1.agent_id} inventa: '{word1}'")
        print(f"💭 {agent2.agent_id} inventa: '{word2}'")
        
        # Calcular payoff
        payoff = 100 if word1 == word2 else -50
        success_emoji = "🎉" if payoff > 0 else "💥"
        
        print(f"{success_emoji} RESULTADO: {payoff} puntos para cada uno")
        
        # Actualizar vocabulario global
        self.global_vocabulary.add(word1)
        self.global_vocabulary.add(word2)
        self.word_frequency[word1] += 1
        self.word_frequency[word2] += 1
        
        # Actualizar memoria de ambos agentes
        agent1.update_memory(word1, word2, payoff)
        agent2.update_memory(word2, word1, payoff)
        
        # Registrar interacción
        interaction = {
            'round': round_num,
            'agent1_id': agent1.agent_id,
            'agent2_id': agent2.agent_id,
            'word1': word1,
            'word2': word2,
            'payoff': payoff,
            'success': payoff > 0
        }
        self.interaction_history.append(interaction)
        
        return word1, word2, payoff
    
    def analyze_current_state(self) -> Dict:
        """Analiza el estado actual del experimento"""
        if not self.interaction_history:
            return {}
        
        # Palabras más frecuentes
        top_words = dict(Counter(self.word_frequency).most_common(5))
        
        # Tasa de éxito reciente
        recent_interactions = self.interaction_history[-10:] if len(self.interaction_history) >= 10 else self.interaction_history
        success_rate = sum(1 for i in recent_interactions if i['success']) / len(recent_interactions)
        
        # Diversidad vocabulario
        vocab_size = len(self.global_vocabulary)
        
        return {
            'top_words': top_words,
            'success_rate': success_rate,
            'vocab_size': vocab_size,
            'total_interactions': len(self.interaction_history)
        }
    
    def print_status_report(self, round_num: int):
        """Imprime reporte del estado actual"""
        state = self.analyze_current_state()
        if not state:
            return
            
        print(f"\n📊 REPORTE DE ESTADO - RONDA {round_num}")
        print("=" * 50)
        print(f"📈 Tasa de éxito reciente: {state['success_rate']:.1%}")
        print(f"📚 Palabras únicas inventadas: {state['vocab_size']}")
        print(f"🏆 Palabras más populares:")
        for word, freq in state['top_words'].items():
            print(f"   • '{word}': {freq} veces")
        
        # Puntuaciones de agentes
        print(f"\n🎯 PUNTUACIONES ACTUALES:")
        for agent in sorted(self.agents, key=lambda a: a.score, reverse=True):
            print(f"   • {agent.agent_id}: {agent.score} puntos")
    
    def run_simulation(self, n_rounds: int = 50) -> Dict:
        """Ejecuta la simulación completa"""
        print(f"\n🚀 INICIANDO SIMULACIÓN DE {n_rounds} RONDAS")
        print(f"🎯 Los agentes deben inventar palabras para: '{self.target_object}'")
        print("\n" + "="*80)
        
        for round_num in range(1, n_rounds + 1):
            # Ejecutar interacción
            word1, word2, payoff = self.run_interaction(round_num)
            
            # Reporte cada 10 rondas
            if round_num % 10 == 0:
                self.print_status_report(round_num)
            
            # Pausa para no sobrecargar
            time.sleep(0.5)
        
        # Análisis final
        print(f"\n🏁 SIMULACIÓN COMPLETADA")
        print("="*80)
        self.print_final_analysis()
        
        return self.analyze_final_results()
    
    def print_final_analysis(self):
        """Imprime análisis final detallado"""
        state = self.analyze_current_state()
        
        print(f"\n🎊 ANÁLISIS FINAL")
        print("="*50)
        print(f"🎯 Objeto estudiado: {self.target_object}")
        print(f"🔢 Total de interacciones: {state['total_interactions']}")
        print(f"📈 Tasa de éxito final: {state['success_rate']:.1%}")
        print(f"🆕 Palabras únicas inventadas: {state['vocab_size']}")
        
        print(f"\n🏆 RANKING DE PALABRAS INVENTADAS:")
        for i, (word, freq) in enumerate(state['top_words'].items(), 1):
            percentage = (freq / sum(state['top_words'].values())) * 100
            print(f"   {i}. '{word}': {freq} usos ({percentage:.1f}%)")
        
        # ¿Emergió consenso?
        if state['top_words']:
            dominant_word = list(state['top_words'].keys())[0]
            dominant_freq = list(state['top_words'].values())[0]
            dominance = (dominant_freq / sum(state['top_words'].values())) * 100
            
            if dominance > 50:
                print(f"\n✅ ¡CONSENSO EMERGENTE! La palabra '{dominant_word}' domina ({dominance:.1f}%)")
            else:
                print(f"\n🔄 SIN CONSENSO CLARO. Múltiples palabras compiten")
        
        print(f"\n👥 PUNTUACIONES FINALES:")
        for agent in sorted(self.agents, key=lambda a: a.score, reverse=True):
            print(f"   • {agent.agent_id}: {agent.score} puntos")
            
        # Innovación vs Imitación
        print(f"\n🧬 ANÁLISIS DE INNOVACIÓN:")
        total_words = sum(self.word_frequency.values())
        unique_words = len(self.global_vocabulary)
        innovation_rate = unique_words / total_words
        print(f"   • Tasa de innovación: {innovation_rate:.2f}")
        print(f"   • Diversidad lingüística: {unique_words} palabras únicas")
        
        if innovation_rate > 0.7:
            print("   🎨 Alta creatividad - Los agentes siguen innovando")
        elif innovation_rate > 0.4:
            print("   ⚖️  Balance entre innovación y convergencia")
        else:
            print("   🎯 Convergencia fuerte - Los agentes imitan palabras exitosas")
    
    def analyze_final_results(self) -> Dict:
        """Retorna análisis final para procesamiento posterior"""
        state = self.analyze_current_state()
        
        return {
            'target_object': self.target_object,
            'final_success_rate': state['success_rate'],
            'vocabulary_size': state['vocab_size'],
            'word_frequencies': dict(self.word_frequency),
            'top_words': state['top_words'],
            'agent_scores': {agent.agent_id: agent.score for agent in self.agents},
            'total_interactions': len(self.interaction_history),
            'interaction_history': self.interaction_history
        }

def main():
    """Función principal para ejecutar experimentos creativos"""
    
    print("🎮 JUEGO DE NOMBRES CREATIVO - INVENCIÓN DE PALABRAS")
    print("🧬 Los agentes LLM deben inventar palabras para objetos misteriosos")
    print("=" * 80)
    
    # Objetos misteriosos para experimentar
    mysterious_objects = [
        "una criatura extraña con tentáculos brillantes y ojos múltiples",
        "un dispositivo tecnológico que emite luz azul y hace ruidos extraños", 
        "una planta alienígena con hojas que cambian de color",
        "un cristal flotante que vibra y produce melodías",
        "una herramienta antigua con símbolos desconocidos"
    ]
    
    # Seleccionar objeto aleatorio
    selected_object = random.choice(mysterious_objects)
    
    print(f"\n🎯 OBJETO SELECCIONADO: {selected_object}")
    print(f"🤖 Verificando conexión con Ollama...")
    
    # Verificar Ollama
    try:
        test_agent = InventiveOllamaAgent("llama3")
        test_response = test_agent._make_ollama_request("Test connection")
        print("✅ Conexión con Ollama exitosa!")
    except Exception as e:
        print(f"❌ Error conectando con Ollama: {e}")
        print("Asegúrate de que Ollama esté corriendo: ollama serve")
        return
    
    # Configuración del experimento
    config = {
        'n_agents': 6,
        'n_rounds': 200,  # Reducido como solicitaste
        'target_object': selected_object,
        'model_name': 'llama3'
    }
    
    print(f"\n⚙️  CONFIGURACIÓN:")
    print(f"   • Agentes: {config['n_agents']}")
    print(f"   • Rondas: {config['n_rounds']}")
    print(f"   • Modelo: {config['model_name']}")
    
    # Crear y ejecutar simulación
    simulation = CreativaNamingGameSimulation(
        n_agents=config['n_agents'],
        target_object=config['target_object'],
        model_name=config['model_name']
    )
    
    # Ejecutar experimento
    results = simulation.run_simulation(n_rounds=config['n_rounds'])
    
    # Guardar resultados
    with open('creative_naming_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: creative_naming_results.json")
    print(f"🎉 ¡Experimento completado!")

if __name__ == "__main__":
    main()