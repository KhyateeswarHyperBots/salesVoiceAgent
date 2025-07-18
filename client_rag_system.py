#!/usr/bin/env python3
"""
Client RAG System for Sales Voice Agent
Integrates clients.json with web search for real-time client context
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple
import pickle

class ClientRAGSystem:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.clients = []
        self.client_embeddings = []
        
        # File paths
        self.clients_file = "clients.json"
        self.client_index_path = "client_vector.index"
        self.client_embeddings_path = "client_embeddings.pkl"
        
    def load_clients(self):
        """Load clients from clients.json"""
        try:
            with open(self.clients_file, 'r') as f:
                self.clients = json.load(f)
            print(f"✅ Loaded {len(self.clients)} clients from {self.clients_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading clients: {e}")
            return False
    
    def build_client_index(self):
        """Build FAISS index for client search"""
        if not self.clients:
            print("❌ No clients loaded")
            return False
        
        print("🔨 Building client search index...")
        
        # Create embeddings for each client
        client_texts = []
        for client in self.clients:
            # Create rich text representation for each client
            text = self._create_client_text(client)
            client_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(client_texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index and embeddings
        faiss.write_index(self.index, self.client_index_path)
        with open(self.client_embeddings_path, 'wb') as f:
            pickle.dump({
                'clients': self.clients,
                'embeddings': embeddings,
                'texts': client_texts
            }, f)
        
        print(f"✅ Built client index with {len(self.clients)} clients")
        return True
    
    def load_client_index(self):
        """Load existing client index"""
        try:
            self.index = faiss.read_index(self.client_index_path)
            with open(self.client_embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.clients = data['clients']
                self.client_embeddings = data['embeddings']
            print(f"✅ Loaded client index with {len(self.clients)} clients")
            return True
        except Exception as e:
            print(f"❌ Error loading client index: {e}")
            return False
    
    def _create_client_text(self, client: Dict) -> str:
        """Create rich text representation of a client"""
        parts = []
        
        # Basic info
        parts.append(f"Name: {client.get('Full Name', 'Unknown')}")
        parts.append(f"Title: {client.get('Title', 'Unknown')}")
        parts.append(f"Company: {client.get('Company', 'Unknown')}")
        parts.append(f"Email: {client.get('Email', 'Unknown')}")
        parts.append(f"Phone: {client.get('Phone', 'Unknown')}")
        
        # Location
        location_parts = []
        if client.get('City'):
            location_parts.append(client['City'])
        if client.get('State'):
            location_parts.append(client['State'])
        if client.get('Country'):
            location_parts.append(client['Country'])
        
        if location_parts:
            parts.append(f"Location: {', '.join(location_parts)}")
        
        # Domain/Website
        if client.get('Domain'):
            parts.append(f"Website: {client['Domain']}")
        
        # LinkedIn
        if client.get('LinkedIn'):
            parts.append(f"LinkedIn: {client['LinkedIn']}")
        
        return " | ".join(parts)
    
    def search_clients(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for clients based on query"""
        if self.index is None:
            print("❌ Client index not loaded")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search index
        D, I = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return matching clients
        results = []
        for idx in I[0]:
            if idx < len(self.clients):
                client = self.clients[idx].copy()
                client['similarity_score'] = float(D[0][list(I[0]).index(idx)])
                results.append(client)
        
        return results
    
    def search_client_by_phone(self, phone_number: str) -> Optional[Dict]:
        """Search for client by phone number"""
        if not self.clients:
            print("❌ No clients loaded")
            return None
        
        # Clean phone number (remove spaces, dashes, etc.)
        clean_phone = ''.join(filter(str.isdigit, phone_number))
        
        # Search for exact match
        for client in self.clients:
            client_phone = client.get('Phone', '')
            if client_phone:
                clean_client_phone = ''.join(filter(str.isdigit, client_phone))
                if clean_phone in clean_client_phone or clean_client_phone in clean_phone:
                    print(f"✅ Found client by phone: {client.get('Full Name')}")
                    return client
        
        # If no exact match, try partial match
        for client in self.clients:
            client_phone = client.get('Phone', '')
            if client_phone and phone_number in client_phone:
                print(f"✅ Found client by partial phone match: {client.get('Full Name')}")
                return client
        
        print(f"❌ No client found with phone number: {phone_number}")
        return None
    
    def identify_client(self, user_input: str) -> Optional[Dict]:
        """Identify client from user input"""
        # Try to extract name or company from user input
        search_terms = self._extract_search_terms(user_input)
        
        if not search_terms:
            return None
        
        # Search for matching clients
        for term in search_terms:
            results = self.search_clients(term, top_k=3)
            if results and results[0]['similarity_score'] > 0.7:  # High confidence match
                return results[0]
        
        return None
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract potential client search terms from text"""
        terms = []
        
        # Look for names (2-3 word patterns)
        words = text.split()
        for i in range(len(words) - 1):
            # Check for name patterns
            if words[i][0].isupper() and words[i+1][0].isupper():
                name = f"{words[i]} {words[i+1]}"
                terms.append(name)
                
                # Check for 3-word names
                if i < len(words) - 2 and words[i+2][0].isupper():
                    name = f"{words[i]} {words[i+1]} {words[i+2]}"
                    terms.append(name)
        
        # Look for company names (capitalized words)
        for word in words:
            if word[0].isupper() and len(word) > 2:
                terms.append(word)
        
        return list(set(terms))  # Remove duplicates
    
    def get_client_context(self, client: Dict) -> str:
        """Get comprehensive context for a client"""
        context_parts = []
        
        # Basic client info
        context_parts.append(f"Client: {client.get('Full Name', 'Unknown')}")
        context_parts.append(f"Title: {client.get('Title', 'Unknown')}")
        context_parts.append(f"Company: {client.get('Company', 'Unknown')}")
        context_parts.append(f"Location: {client.get('City', '')}, {client.get('State', '')}, {client.get('Country', '')}")
        
        return "\n".join(context_parts)
    
    def enhance_conversation_with_client_context(self, user_input: str, conversation_history: List[Dict] = None) -> Tuple[str, Optional[Dict]]:
        """Enhance conversation with client context"""
        # Try to identify client
        client = self.identify_client(user_input)
        
        if client:
            print(f"👤 Identified client: {client.get('Full Name')} from {client.get('Company')}")
            
            # Get comprehensive context
            context = self.get_client_context(client)
            
            # Create enhanced prompt
            enhanced_prompt = f"""
Client Context:
{context}

User Input: {user_input}

Please provide a personalized response considering the client's role, company, and recent information.
"""
            return enhanced_prompt, client
        else:
            # No client identified, return original input
            return user_input, None
    
    def save_client_interaction(self, client: Dict, interaction: Dict):
        """Save client interaction for future reference"""
        os.makedirs("client_interactions", exist_ok=True)
        
        client_id = client.get('Email', client.get('Full Name', 'unknown'))
        filename = f"client_interactions/{client_id.replace('@', '_').replace('.', '_')}.json"
        
        # Load existing interactions or create new
        interactions = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                interactions = json.load(f)
        
        # Add new interaction
        interaction['timestamp'] = interaction.get('timestamp', '')
        interactions.append(interaction)
        
        # Save updated interactions
        with open(filename, 'w') as f:
            json.dump(interactions, f, indent=2)
        
        print(f"💾 Saved interaction for {client.get('Full Name')}")
    
    def get_client_summary(self, client: Dict) -> str:
        """Get a summary of client information"""
        summary_parts = []
        
        summary_parts.append(f"👤 {client.get('Full Name', 'Unknown')}")
        summary_parts.append(f"🏢 {client.get('Company', 'Unknown')}")
        summary_parts.append(f"💼 {client.get('Title', 'Unknown')}")
        summary_parts.append(f"📍 {client.get('City', '')}, {client.get('State', '')}")
        summary_parts.append(f"📧 {client.get('Email', 'Unknown')}")
        summary_parts.append(f"📞 {client.get('Phone', 'Unknown')}")
        
        return "\n".join(summary_parts)

def main():
    """Test the client RAG system"""
    client_rag = ClientRAGSystem()
    
    # Load clients
    if not client_rag.load_clients():
        return
    
    # Build or load index
    if not os.path.exists(client_rag.client_index_path):
        client_rag.build_client_index()
    else:
        client_rag.load_client_index()
    
    # Test client search
    print("\n🧪 Testing Client Search")
    print("=" * 40)
    
    test_queries = [
        "Sam Grizzle",
        "CFO",
        "Vanguard Truck Centers",
        "Atlanta finance"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = client_rag.search_clients(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('Full Name')} - {result.get('Title')} at {result.get('Company')} (Score: {result['similarity_score']:.3f})")
    
    # Test client identification
    print(f"\n🧪 Testing Client Identification")
    print("=" * 40)
    
    test_inputs = [
        "Hi, I'm Sam Grizzle from TBS Brands",
        "I'm the CFO at Vanguard Truck Centers",
        "Can you tell me about your AI solutions?"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: '{user_input}'")
        enhanced_prompt, client = client_rag.enhance_conversation_with_client_context(user_input)
        
        if client:
            print(f"✅ Identified: {client.get('Full Name')} from {client.get('Company')}")
            print(f"📝 Enhanced context length: {len(enhanced_prompt)} characters")
        else:
            print("❌ No client identified")

if __name__ == "__main__":
    main() 