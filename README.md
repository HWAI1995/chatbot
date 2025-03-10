# Inhaltsverzeichnis

1. [Ziel](#ziel)
2. [Was ist generative AI?](#was-ist-generative-ai)
3. [Einbettung und Tokenisierung](#einbettung-und-tokenisierung)
4. [Wie erweitert man ein LLM mit externer Wissensbasis?](#wie-erweitert-man-ein-llm-mit-externer-wissensbasis)
5. [Architektur hinter einem LLM](#architektur-hinter-einem-llm)
6. [Prompt Engineering](#prompt-engineering)
7. [Welche Tools werden benötigt?](#welche-tools-werden-benötigt)
8. [Langchain vs OpenAI Assistant](#langchain-vs-openai-assistant)

## Ziel

Das Ziel ist, Kunden beim Support durch einen Chatbot zu unterstützen. Wenn ein Kunde Fragen zur Fakturierung hat, kann er den Chatbot nutzen, der ihm gezielt weiterhilft. Der Chatbot soll dabei hilfreiche Antworten liefern und die passende Quelle im Dokument angeben. Dadurch spart der Kunde Zeit, da er nicht selbst nach Informationen suchen oder den Kundensupport kontaktieren muss.

## Was ist generative AI?

Unter generative AI versteht man KI, die Inhalte aufgrund Benutzereingaben generiert. Diese Modelle nutzen neuronale Netze, um Muster aus neuen Daten zu erkennen und Sprache in Einbettungen sogenannte Embeddings zu konvertieren.  
Man unterteilt darunter folgende Anwendungsbereiche:

- Textherstellung
- Bildgenerierung
- Musik und Audio
- Videoerstellung

Die Vorteile von Gen AI ist die Automatisierung von Routineaufgaben und die Generierung von kreativen Inhalten.  
Der Schlüssel zu Generative AI sind Einbettungen, die auch nachher erläutert werden.

## Einbettung und Tokenisierung

Zuallererst muss man verstehen, dass KI keine Wörter oder Strings versteht, sondern lediglich Tokens, weil es schwierig für eine KI ist, Strings eine semantische Bedeutung zu geben. Das heißt, es muss gezielt werden Wörter in etwas, um zu wandeln, womit Systeme arbeiten können, und dazu betrachtet man Tokens.  
Unter einer Tokenisierung versteht man im Grunde genommen die Aufteilung eines Textes in die kleinste Einheit. Ein Token kann ein Satz, Wort oder ein Satzzeichen sein.

**Wort-Tokenisierung Beispiel:**

Eingabe: „Ich mag KI!“ wird in dem Fall „tokenisiert“ als [„Ich“, „mag“, „KI“]

**Satz-Tokenisierung Beispiel:**

„Hallo, wie geht es dir? Ich heiße Lars.“ wird „tokenisiert“ zu [„Hallo!", „wie geht es dir?", „Ich heiße Lars"]

Diese Tokens werden indexiert und mit einer ID versehen, die zur Bearbeitung von LLMs entscheidend ist.  
Es gibt verschiedene Möglichkeiten der Tokenisierung. Einbettungen bestehen im Wesentlichen aus Vektordarstellungen einzelner Tokens. Diese ermöglichen eine semantische und kontextuelle Repräsentation und werden von speziellen KI-Modellen generiert.

## Wie erweitert man ein LLM mit externer Wissensbasis?

Einer der größten Anwendungsbereiche in LLM liegt in der Erweiterung mit externen Daten. Dadurch entwickeln wir Systeme, die Zugriff auf Dokumente haben.  
Die Technik dazu nennt sich RAG (Retrieval augmented generation).

**Retrieval Augmented Generation (RAG):**

RAG verbessert Sprachmodelle durch Integration externer Wissensquellen. Es sucht in einer Wissensdatenbank nach relevanten Informationen und fügt diese in den Prompt ein, wodurch genauere Antworten generiert werden. Diese Technik macht KI-Systeme leistungsfähiger und zuverlässiger.

**Abbildung 1**: Schlüsselkonzepte von RAG

1. Abruf relevanter Informationen in der Vektordatenbank
2. Weitergabe der Informationen mit dem Prompt an das Modell
3. Verarbeitung und Ausgabe des Ergebnisses an den Nutzer

## Architektur hinter einem LLM

Ein Large Language Model (LLM) versteht und generiert natürliche Sprache mittels neuronaler Netze, insbesondere der Transformer-Architektur. Die Trainingsphasen sind Pretraining und Feintuning.

- **Pretraining**: Das Modell lernt enormen Textmengen, um das nächste Wort in einem Satz vorherzusagen.
- **Feintuning**: Optimierung für spezifische Aufgaben, z.B. durch Reinforcement Learning mit menschlichem Feedback (RLHF).

Ein LLM nutzt den Self-Attention-Mechanismus, um Bedeutungen einzelner Wörter im Kontext zu analysieren, und wendet Wahrscheinlichkeitsberechnungen an, um Wortfolgen zu bestimmen.

Einsatzmöglichkeiten:
- Chatbots in Kundenservice-Systemen
- Automatische Texterstellung
- Programmcode schreiben
- Übersetzungen
- Zusammenfassungen erstellen

## Prompt Engineering

Prompt Engineering bezeichnet die gezielte Formulierung von Eingaben, um optimale Antworten von KI-Systemen zu erhalten. Die Qualität und Struktur des Prompts beeinflussen die Qualität der Ausgabe.

### Wichtige Techniken:

- **Klarheit und Präzision**: Prompts sollten kurz und genau sein.
  - Schlecht: "Erkläre mir was über LLMs"
  - Besser: "Erkläre mir die Grundlagen zu LLMs in einfachen Worten"

- **Rollenbasierte Anweisungen**: Der KI kann mitgeteilt werden, als was sie agieren soll. 

- **Negative Prompting**: Unerwünschte Ergebnisse explizit angeben.

Weitere Techniken sind auf der Plattform von OpenAi zu finden.

## Welche Tools werden benötigt?

Zur Entwicklung eines KI-Prototypen benötigt man:

- Ein LLM, das Eingabe verarbeitet und auf eine Vektordatenbank zugreifen kann
- Eine Vektordatenbank, z.B. MongoDB, zum Speichern und Persistieren der Daten
- Ein Framework zur Orchestrierung, Verwaltung und Integration der Systeme

Um ein LLM zu nutzen, benötigt man eine Open AI Plus Version (20 Euro/Nutzer) und entsprechende Credits für die API. MongoDB bietet kostenlose Speicherung bis 512MB. Die Orchestrierung kann durch Langchain oder OpenAI Assistants erfolgen.

### Tools:

1. **LangChain**: Open-Source-Bibliothek für KI-Anwendungen mit LLMs
2. **OpenAI Assistant**: API für benutzerdefinierte KI-Assistenten mit OpenAI-Modellen

## Langchain vs OpenAI Assistant 

### LangChain:

- **Vorteile**:
  - Flexibilität bei verschiedenen LLMs
  - Integration von Vektordatenbanken
  - Erweiterbarkeit mit externen Tools
  - Offene Architektur für Anpassungen
  
- **Nachteile**:
  - Höhere Komplexität
  - Längere Entwicklungszeiten 
  - Potenzielle Latenz

### OpenAI Assistants API:

- **Vorteile**:
  - Einfache Nutzung ohne komplexe Infrastruktur
  - Integrierte Funktionen 
  - Optimierung durch OpenAI

- **Nachteile**:
  - Weniger Flexibilität (nur OpenAI-Modelle)
  - Kostenpotenzial bei vielen API-Calls
  - Begrenzte Anpassungsmöglichkeiten 

### Fazit:

- **LangChain**: Geeignet für komplexe, maßgeschneiderte Anwendungen.
- **OpenAI Assistants API**: Ideal für schnelle, skalierbare Implementationen.

Für einfache Anwendungsfälle oder schnelle Prototypen ist die OpenAI Assistants API die beste Wahl. Für eine hochgradig anpassbare Lösung ist LangChain zu bevorzugen.# projekt
