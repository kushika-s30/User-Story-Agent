from __future__ import annotations # Enables forward references for type hints (using a class as a type within its own definition)
import os # Provides functions for interacting with the operating system (e.g., reading environment variables)
import json # Library for parsing and creating JSON data
import argparse # Tool for writing user-friendly command-line interfaces
import requests # HTTP library for making API requests to external services like Trello
from typing import List, Optional # Imports type hinting generics for lists and optional (nullable) values
from pydantic import BaseModel, Field # Core classes for data validation and schema definition using Python classes
from langchain_core.messages import SystemMessage, HumanMessage # Classes to represent different roles in a conversation with an LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Specialized wrappers for OpenAI's chat models and vector embedding models
from dotenv import load_dotenv # Utility to load environment variables from a .env file into the system's environment

# RAG Imports for handling document retrieval and text splitting
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (API keys) from a .env file
load_dotenv(override=True)

# Global Configuration for Trello API access
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")


# --- DATA MODELS (Pydantic) ---
# These classes define the structure the AI must follow when generating JSON output.

class SubTask(BaseModel):
    """Schema for a single checklist item in a user story."""
    task: str = Field(description="The sub-task description")


class AcceptanceCriterion(BaseModel):
    """Schema for a single 'Definition of Done' point."""
    criterion: str = Field(description="The acceptance criterion description")


class UserStory(BaseModel):
    """Main schema for an Agile User Story, linking criteria and tasks together."""
    user_story: str = Field(description="The user story in format: As a [user], I want [intent], so that [value].")
    acceptance_criteria: List[AcceptanceCriterion] = Field(description="List of acceptance criteria")
    sub_tasks: List[SubTask] = Field(description="List of sub-tasks")


class UserStoryCollection(BaseModel):
    """A wrapper class to allow the AI to return a list of multiple stories at once."""
    stories: List[UserStory] = Field(description="List of user stories extracted from the transcript")


def load_system_prompt() -> str:
    """Reads the 'brain' instructions for the AI from an external text file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "PromptTemplate.txt")
    try:
        with open(prompt_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"PromptTemplate.txt not found. Please ensure it is in the same directory."
        )


def create_trello_card(user_story: UserStory, list_id: str) -> Optional[str]:
    """
    Automates the creation of a Trello card. 
    1. Creates the card with story details.
    2. Adds a 'Sub-Tasks' checklist based on the AI's output.
    """
    if not TRELLO_API_KEY or not TRELLO_TOKEN:
        print("Warning: Missing Trello credentials. Skipping card creation.")
        return None
    
    # Format description using only acceptance criteria for the card body
    description = "## Acceptance Criteria\n\n"
    for criterion in user_story.acceptance_criteria:
        description += f"- {criterion.criterion}\n"

    # API endpoint for card creation
    url = "https://api.trello.com/1/cards"
    query = {
        'idList': list_id,
        'key': TRELLO_API_KEY,
        'token': TRELLO_TOKEN,
        'name': user_story.user_story,
        'desc': description
    }
    
    try:
        response = requests.post(url, params=query)
        if response.status_code == 200:
            card_id = response.json().get('id')
            print(f"✅ Created Trello card: {user_story.user_story[:40]}...")
            
            # --- Native Checklist Creation ---
            # Create a checklist named "Sub-Tasks" inside the newly created card
            checklist_url = f"https://api.trello.com/1/cards/{card_id}/checklists"
            checklist_query = {
                'key': TRELLO_API_KEY,
                'token': TRELLO_TOKEN,
                'name': 'Sub-Tasks'
            }
            
            checklist_resp = requests.post(checklist_url, params=checklist_query)
            if checklist_resp.status_code == 200:
                checklist_id = checklist_resp.json().get('id')
                
                # Add each AI-generated sub-task as a checklist item
                for task in user_story.sub_tasks:
                    item_url = f"https://api.trello.com/1/checklists/{checklist_id}/checkItems"
                    item_query = {
                        'key': TRELLO_API_KEY,
                        'token': TRELLO_TOKEN,
                        'name': task.task
                    }
                    requests.post(item_url, params=item_query)
                print(f"   ✅ Added checklist with {len(user_story.sub_tasks)} sub-tasks")
                
            return card_id
        else:
            print(f"❌ Trello API Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating Trello card: {e}")
        return None


def format_user_story_output(story: UserStory, index: int) -> str:
    """Formats the UserStory object into a readable block for console or file logging."""
    output = f"\n{'='*80}\n"
    output += f"USER STORY #{index}\n"
    output += f"{'='*80}\n\n"
    output += f"**User Story:**\n{story.user_story}\n\n"
    output += "**Acceptance Criteria:**\n"
    for criterion in story.acceptance_criteria:
        output += f"- {criterion.criterion}\n"
    output += "\n**Sub-Tasks:**\n"
    for task in story.sub_tasks:
        output += f"- [ ] {task.task}\n"
    return output


def process_transcript(transcript: str, trello_list_id: Optional[str] = None, 
                       output_file: Optional[str] = None) -> UserStoryCollection:
    """
    Main orchestration function: 
    1. Loads the prompt.
    2. Sends transcript to GPT-4o-mini with a strict output schema.
    3. Handles Trello syncing and local file saving.
    """
    system_prompt = load_system_prompt()
    
    # Initialize the LLM. 'gpt-4o-mini' is used for cost-effective reasoning.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0, # Temperature 0 ensures deterministic, focused responses
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 'with_structured_output' forces the model to return valid JSON matching the Pydantic class
    structured_llm = llm.with_structured_output(UserStoryCollection)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Convert this transcript into user stories:\n\n{transcript}")
    ]
    
    print("🔄 Processing transcript with OpenAI...")
    result = structured_llm.invoke(messages)
    
    print(f"✅ Generated {len(result.stories)} user story(ies)\n")
    
    full_output = ""
    for i, story in enumerate(result.stories, 1):
        story_output = format_user_story_output(story, i)
        full_output += story_output
        print(story_output)
        
        # Trigger Trello integration if an ID was provided
        if trello_list_id:
            create_trello_card(story, trello_list_id)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\n💾 Output saved to: {output_file}")
    
    return result


class RAGPipeline:
    """
    Handles 'Question-Answering' mode. 
    It 'reads' the transcript and indexes it so you can ask specific questions.
    """
    def __init__(self, text_content: str):
        print("🔄 Initializing RAG pipeline (Indexing transcript)...")
        # Split text into chunks to fit into the AI's context window effectively
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.create_documents([text_content])
        
        # Create numerical embeddings (vectors) of the text
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Store vectors in a searchable FAISS database
        self.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        print("✅ RAG pipeline ready.")

    def ask(self, query: str) -> dict:
        """Finds the most relevant chunks in the transcript and answers the question."""
        # Retrieve the 3 most relevant segments from the vector store
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            "You are an assistant. Answer the question using ONLY the provided context. "
            "Keep answers concise (max 3 sentences)."
            "\n\n"
            f"Context:\n{context}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "context": [d.page_content for d in docs]
        }

def setup_rag_pipeline(text_content: str) -> RAGPipeline:
    return RAGPipeline(text_content)


def main():
    """CLI Entry point. Handles user commands from the terminal."""
    parser = argparse.ArgumentParser(description="Transcript to User Story Agent")
    
    # Input handling: provide a file (-i) or raw text (-t)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str, help='Path to transcript file')
    input_group.add_argument('--text', '-t', type=str, help='Direct transcript text')
    
    # Trello & Output Flags
    parser.add_argument('--trello-list-id', '-l', type=str, help='Trello list ID')
    parser.add_argument('--no-trello', action='store_true', help='Disable Trello sync')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    
    # Query/Chat Flags
    parser.add_argument('--query', '-q', type=str, help='Specific question about transcript')
    parser.add_argument('--chat', action='store_true', help='Enter interactive chat mode')
    
    args = parser.parse_args()
    
    # API Key check
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found.")
        return
    
    # Load transcript data
    if args.input:
        with open(args.input, 'r') as f:
            transcript = f.read()
    else:
        transcript = args.text
    
    # --- Execute RAG Path (Chat/Query) ---
    if args.query or args.chat:
        rag = setup_rag_pipeline(transcript)
        
        if args.query:
            answer = rag.ask(args.query)
            print(f"\n💡 Answer: {answer['answer']}\n")
            
        if args.chat:
            print("\n💬 Entering chat mode (type 'exit' to quit):")
            while True:
                user_input = input("\n> ")
                if user_input.lower() in ["exit", "quit"]: break
                answer = rag.ask(user_input)
                print(f"💡 Answer: {answer['answer']}")
        return 

    # --- Execute Standard Path (Story Generation) ---
    trello_list_id = None if args.no_trello else (args.trello_list_id or os.getenv("TRELLO_LIST_ID"))
    
    try:
        process_transcript(transcript, trello_list_id, args.output)
        print("\n✅ All operations complete!")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")


if __name__ == "__main__":
    main()
