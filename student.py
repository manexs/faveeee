from flask import Flask, render_template, request, jsonify, redirect, url_for,session
import pandas as pd
from openai import OpenAI
import os
import threading
import time
import traceback
from dotenv import load_dotenv
import sqlite3
import json
from flask import jsonify, request
import traceback 
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import json
import uuid
from urllib.parse import quote
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import io
import base64
import os
import uuid  
from werkzeug.utils import secure_filename
import matplotlib
from datetime import datetime
import re  
matplotlib.use('Agg')  # non-interactive backend


try:
    # MongoDB Connection
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['tarp']
    print("Connected to MongoDB successfully")
    
    # Create necessary collections if they don't exist
    if 'events' not in db.list_collection_names():
        db.create_collection('events')
    
    if 'messages' not in db.list_collection_names():
        db.create_collection('messages')
        # Create indexes for performance
        db.messages.create_index([("to_user_id", 1)])
        db.messages.create_index([("from_user_id", 1)])
        db.messages.create_index([("event_id", 1)])
        db.messages.create_index([("is_read", 1)])
        
    if 'users' not in db.list_collection_names():
        db.create_collection('users')
        db.users.create_index([("user_id", 1)], unique=True)
        
    print("MongoDB database setup complete")
    
    # Set a flag to indicate we're using MongoDB
    using_mongodb = True
    
except Exception as e:
    print(f"MongoDB connection error: {e}")
    # Fallback to SQLite if MongoDB connection fails
    print("Falling back to SQLite...")
    using_mongodb = False


class ConversationContextManager:
    def __init__(self, db):
        self.db = db
    
    def get_active_context(self, user_id):
        """Get the active conversation context for a user"""
        user_doc = self.db.users.find_one({'user_id': user_id})
        if not user_doc or 'active_context' not in user_doc:
            return None
            
        context = user_doc['active_context']
        
        # Check if context has expired
        if 'expires_at' in context:
            expires_at = datetime.strptime(context['expires_at'], "%Y-%m-%d %H:%M:%S")
            if datetime.now() > expires_at:
                # Context expired, clear it
                self.clear_context(user_id)
                return None
                
        return context
    
    def set_context(self, user_id, context_type, context_data, expiry_hours=24):
        """Set active conversation context for a user"""
        context = {
            'type': context_type,
            'data': context_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expires_at': (datetime.now() + timedelta(hours=expiry_hours)).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.db.users.update_one(
            {'user_id': user_id},
            {'$set': {'active_context': context}},
            upsert=True
        )
        
    def clear_context(self, user_id):
        """Clear the active conversation context for a user"""
        self.db.users.update_one(
            {'user_id': user_id},
            {'$unset': {'active_context': 1}}
        )

# Then, initialize the context manager after the MongoDB connection is confirmed successful


app = Flask(__name__)
app.secret_key = 'a1b2c3d4e5f6g7h8i9j0aabbccddeeff112233445566778899'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()  # This loads the variables from a .env file

# Configure OpenRouter client
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
site_url = os.getenv("SITE_URL", "http://localhost:5000")
site_name = os.getenv("SITE_NAME", "Social Connection Platform")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# Updated interest categories for real-world users
INTEREST_CATEGORIES = {
    'Technology': ['Programming', 'Web Development', 'AI', 'Blockchain', 'Cybersecurity', 'Mobile Apps', 'Gaming', 'Tech Gadgets'],
    'Arts': ['Photography', 'Painting', 'Drawing', 'Sculpture', 'Digital Art', 'Graphic Design', 'Crafts', 'Music Production'],
    'Food': ['Cooking', 'Baking', 'Restaurants', 'Wine', 'Coffee', 'Vegan', 'Barbecue', 'International Cuisine'],
    'Fitness': ['Running', 'Yoga', 'Weight Training', 'Cycling', 'Hiking', 'Swimming', 'CrossFit', 'Martial Arts'],
    'Entertainment': ['Movies', 'TV Shows', 'Music', 'Books', 'Podcasts', 'Concerts', 'Theater', 'Streaming'],
    'Travel': ['Adventure Travel', 'Backpacking', 'Road Trips', 'International Travel', 'Budget Travel', 'Luxury Travel', 'Solo Travel'],
    'Sports': ['Basketball', 'Football', 'Soccer', 'Tennis', 'Golf', 'Baseball', 'Volleyball', 'Esports'],
    'Home': ['Interior Design', 'DIY', 'Gardening', 'Home Renovation', 'Furniture', 'Smart Home', 'Organization', 'Cooking Equipment'],
    'Fashion': ['Streetwear', 'Vintage', 'Luxury', 'Sustainable Fashion', 'Accessories', 'Footwear', 'Personal Style', 'Beauty'],
    'Business': ['Entrepreneurship', 'Investing', 'Marketing', 'Personal Finance', 'Real Estate', 'Startups', 'Side Hustles'],
    'Social': ['Dating', 'Networking', 'Friendship', 'Volunteering', 'Community Service', 'Mentoring', 'Cultural Exchange'],
    'Education': ['Languages', 'History', 'Science', 'Online Courses', 'Self-Improvement', 'Personal Development', 'Life Skills']
}

# Updated events for real-world users
EVENTS = {
    'Technology': ['Hackathons', 'Tech Meetups', 'Programming Workshops', 'Product Launches', 'Gaming Tournaments'],
    'Arts': ['Gallery Openings', 'Art Classes', 'Photography Walks', 'Creative Workshops', 'Music Jams'],
    'Food': ['Cooking Classes', 'Food Festivals', 'Wine Tastings', 'Pop-up Restaurants', 'Farmers Markets'],
    'Fitness': ['Group Workouts', 'Yoga Classes', 'Running Clubs', 'Hiking Trips', 'Fitness Challenges'],
    'Entertainment': ['Movie Screenings', 'Concert Tickets', 'Comedy Shows', 'Book Clubs', 'Live Music'],
    'Travel': ['Travel Meetups', 'Group Trips', 'Adventure Excursions', 'City Tours', 'Cultural Exchanges'],
    'Sports': ['Game Tickets', 'Sports Leagues', 'Watch Parties', 'Amateur Competitions', 'Training Sessions'],
    'Home': ['Furniture Swaps', 'DIY Workshops', 'Home Decor Events', 'Gardening Classes', 'Neighborhood Garage Sales'],
    'Fashion': ['Clothing Swaps', 'Fashion Shows', 'Style Workshops', 'Vintage Markets', 'Designer Meetups'],
    'Business': ['Networking Events', 'Pitch Competitions', 'Workshops', 'Industry Conferences', 'Coworking Meetups'],
    'Social': ['Singles Events', 'Community Service', 'Speed Dating', 'Language Exchanges', 'New in Town Meetups'],
    'Education': ['Workshops', 'Seminars', 'Classes', 'Lectures', 'Study Groups', 'Skill Exchanges'],
    'General': ['Happy Hours', 'Seasonal Festivals', 'Community Gatherings', 'Charity Events', 'Weekend Markets', 'Local Fairs']
}

# Marketplace categories for real-world users
MARKETPLACE_CATEGORIES = {
    'Goods': ['Electronics', 'Furniture', 'Clothing', 'Books', 'Collectibles', 'Art', 'Home & Garden', 'Vehicles'],
    'Services': ['Professional', 'Creative', 'Technical', 'Educational', 'Household', 'Health & Wellness', 'Event-based'],
    'Spaces': ['Living Spaces', 'Work Spaces', 'Storage', 'Event Venues', 'Parking', 'Land', 'Recreational Spaces'],
    'Skills': ['Technical Skills', 'Creative Skills', 'Language Skills', 'Business Skills', 'Academic Knowledge', 'Life Skills'],
    'Experiences': ['Local Tours', 'Workshops', 'Classes', 'Adventures', 'Cultural Experiences', 'Food Experiences']
}

VALID_CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
               "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
               "San Francisco", "Columbus", "Indianapolis", "London", "Tokyo", "Paris", "Berlin"]
# Create flat lists of all interests and events
ALL_INTERESTS = []
for interests in INTEREST_CATEGORIES.values():
    ALL_INTERESTS.extend(interests)
ALL_INTERESTS = list(set(ALL_INTERESTS))  # Remove duplicates

ALL_EVENTS = []
for events in EVENTS.values():
    ALL_EVENTS.extend(events)
ALL_EVENTS = list(set(ALL_EVENTS))  # Remove duplicates

ALL_MARKETPLACE_ITEMS = []
for items in MARKETPLACE_CATEGORIES.values():
    ALL_MARKETPLACE_ITEMS.extend(items)
ALL_MARKETPLACE_ITEMS = list(set(ALL_MARKETPLACE_ITEMS))  # Remove duplicates


# Updated LLM prompt for classifying user queries in a social platform
CLASSIFIER_PROMPT = """
You are an AI designed to analyze and classify user queries on a social platform.
Your only job is to determine what kind of information the user is looking for and extract relevant entities.

CRITICAL: Pay attention to session context. If the user is currently in an event creation flow,
prioritize event_creation classification over event_scheduling.

When you see abbreviations or short forms in the query, convert them to their full forms.
Examples:
- "tech" → "Technology"
- "DIY" → "DIY"

IMPORTANT: Always recognize both verb and noun forms of activities. For example:
- If someone mentions "travelling" or "travels", classify it as "Travel"
- If someone mentions "cooking" or "cooks", classify it as "Cooking"
- If someone mentions "photograph" or "photographing", classify it as "Photography"

OUTPUT FORMAT: You must respond with a JSON object containing:
1. query_type: One of ["interest_search", "event_search", "category_search", "marketplace_search","event_creation","event_scheduling","general_knowledge"]
2. entities: A dictionary containing any of these keys if detected:
   - interests: A LIST of interests mentioned (e.g., ["Photography", "Cooking"]) 
   - event: The specific event mentioned (e.g., "Tech Meetup")
   - category: The specific category mentioned (e.g., "Technology", "Arts")
   - marketplace_item: The specific item mentioned (e.g., "Furniture", "Couch")
   - marketplace_action: Whether user is "looking" or "offering" something
   - general_knowledge: When the user asks factual questions unrelated to the platform (e.g., "how long does it take to get a law degree")

EVENT CREATION vs EVENT SCHEDULING - CRITICAL DISTINCTION:
- event_creation: User wants to CREATE/ORGANIZE a new event
  Examples: "create a soccer event", "I want to organize a meetup", "set up a party"
- event_scheduling: User is RESPONDING to an existing event invitation 
  Examples: "I can attend", "I'm available", "can't make it", "yes I'll be there"

Key indicators for event_creation:
- Words like: "create", "organize", "set up", "plan", "host", "arrange"
- Event type mentioned: "soccer event", "book club", "tech meetup"
- Planning context: "let's do", "I want to", "how about"

Key indicators for event_scheduling:
- Availability responses: "yes", "no", "available", "can't attend"
- Time preferences: "can we do 6pm instead?", "I'm free at 3"
- RSVP language: "I'll be there", "count me in", "I can't make it"

CONTEXT IS CRUCIAL:
- If user is in event creation flow and says "yes" → event_creation (confirmation)
- If user is responding to invitation and says "yes" → event_scheduling (acceptance)
- Simple responses like "yes", "2pm", "dallas" depend heavily on context

Example classifications:
- "Who's interested in Photography?" → {"query_type": "interest_search", "entities": {"interests": ["Photography"]}}
- "Find people for the Tech Meetup" → {"query_type": "event_search", "entities": {"event": "Tech Meetup"}}
- "create a soccer event" → {"query_type": "event_creation", "entities": {"event": "soccer"}}
- "soccer event" → {"query_type": "event_creation", "entities": {"event": "soccer"}}
- "What Technology events are available?" → {"query_type": "event_search", "entities": {"category": "Technology"}}
- "Show me users in the Arts category" → {"query_type": "category_search", "entities": {"category": "Arts"}}
- "I'm looking for people interested in Cooking and Hiking" → {"query_type": "interest_search", "entities": {"interests": ["Cooking", "Hiking"]}}
- "Show me people who like Photography and Graphic Design" → {"query_type": "interest_search", "entities": {"interests": ["Photography", "Graphic Design"]}}
- "Find people interested in AI" → {"query_type": "interest_search", "entities": {"interests": ["AI"]}}
- "Who's into Gardening?" → {"query_type": "interest_search", "entities": {"interests": ["Gardening"]}}
- "Anyone selling furniture?" → {"query_type": "marketplace_search", "entities": {"marketplace_item": "Furniture", "marketplace_action": "offering"}}
- "Looking for a couch" → {"query_type": "marketplace_search", "entities": {"marketplace_item": "Furniture", "marketplace_action": "looking"}}
- "Yes, I can attend the meeting" → {"query_type": "event_scheduling", "entities": {"response": "available"}}
- "Can we do it at 6pm instead?" → {"query_type": "event_scheduling", "entities": {"response": "reschedule", "time": "6pm"}}

IMPORTANT: Respond ONLY with the JSON classification. Do not include any explanations or other text.
"""


# Updated response generator prompt for the social platform
# Updated response generator prompt for the social platform
RESPONSE_GENERATOR_PROMPT = """
You are an AI assistant for a social platform that helps people connect with others who share similar interests and discover relevant events and items.

Your role is to help users find meaningful connections and community. When responding to questions about finding other users or discovering events, use ONLY the data provided to you in the context.

🚨 CRITICAL ANTI-HALLUCINATION RULES:
- You can ONLY recommend users, events, or items that are explicitly provided in the search results data
- If no data is provided or results are empty, you MUST explain that no matches were found
- NEVER create fake users, events, or statistics
- NEVER suggest content that isn't in the actual API response data
- When no results found, acknowledge it clearly and suggest alternatives

Guidelines for your responses:

1. INTEREST RECOMMENDATIONS:
   When recommending users with interests, first check what was ACTUALLY found:
   
   FOR EXACT MATCHES (when matched_interest == searched interest):
   "I found [number] people interested in [interest]! Here are some you might want to connect with:
   
   - User #[ID] ([age], [location])
   - User #[ID] ([age], [location])
   [... up to 5 users ...]
   
   
   FOR FUZZY MATCHES (when matched_interest != searched interest):
   "I couldn't find users specifically interested in [searched_interest], but I found [number] people interested in [matched_interest] which might be related:
   
   - User #[ID] ([age], [location]) - interested in [their actual interest]
   - User #[ID] ([age], [location]) - interested in [their actual interest]
   [... up to 5 users ...]
   
   Would you like me to search for a different interest?"
   
   FOR NO MATCHES:
   "I couldn't find any users interested in [searched_interest] in our current database. You could try searching for related interests or browse our available categories."
   
   IMPORTANT: 
   - Check the 'matched_interest' field in each user result
   - If it differs from the searched interest, acknowledge this is a fuzzy/related match
   - NEVER claim exact matches when the data shows fuzzy matches

2. MULTIPLE INTERESTS:
   When responding to queries about multiple interests:
   - Mention how many users match EACH interest
   - Mention how many users match COMBINATIONS of interests (e.g., "3 people are interested in both Photography and Cooking")
   - Recommend diverse users who match the MOST interests first
   - Example: "I found 12 people interested in Programming, 8 interested in AI, and 3 who are interested in both!"

3. EVENT RECOMMENDATIONS:
   When suggesting events, mention ONLY the actual events from the data provided and why they might be interesting:
   "Here are some events you might enjoy:
   
   - [Event Name]: This would be perfect for meeting other people with similar interests
   - [Event Name]: Many people with similar interests will attend this
   - [Event Name]: This offers great networking opportunities with peers"
   
   🚨 NEVER make up events that aren't in the provided API response data.
   🚨 IF NO EVENTS FOUND: "I don't see any [event type] events in our current database. You could browse our available event categories or create your own event!"

4. CATEGORY QUERIES:
   When responding to category-related queries:
   - For interest questions, list the top interests available in that category
   - For user queries, highlight diverse users from that category
   - If a category isn't found, list the available categories a user can choose from
   - Example: "The Technology category offers these interests: Programming, Web Development, AI..."

5. MARKETPLACE ITEMS:
   When responding to marketplace queries:
   - For "looking" queries, find users offering the item
   - For "offering" queries, find users looking for the item
   - Include relevant details like location for physical items
   - Example: "I found 5 people offering furniture in your area. Here are some listings..."

6. CRITICAL INSTRUCTION ABOUT CONVERSATION MEMORY:
    - You DO have memory of this conversation and previous exchanges with the user
    - NEVER claim that you "don't have memory of previous conversations" or that you "start fresh each time" 
    - If the user refers to something they mentioned before, acknowledge it naturally
    - If you're unsure what they're referring to, ask for clarification rather than claiming you have no memory
    - Maintain continuity in the conversation by referencing previously discussed topics when relevant

7. GENERAL KNOWLEDGE: When the user asks a question unrelated to the social platform, interests, events, or marketplace. Examples:
    - "What's the capital of France?"
    - "How do I bake chocolate chip cookies?"
    - "Tell me about quantum physics"
    - "Who won the World Cup in 2022?"

8. EVENT CREATION:
   When the user wants to create an event:
    - ANY request that mentions an event type (like "soccer event", "create a soccer event", "can you create a soccer event") 
      MUST follow this exact step-by-step process:
      1. First response: Ask ONLY for the DATE
      2. Second response: Acknowledge the date and ask ONLY for the TIME
      3. Third response: Acknowledge the time and ask ONLY for the LOCATION
      4. Fourth response: Show summary of all details and ask for CONFIRMATION
      5. Final response: Only after confirmation, create the event and notify users

🚨 NO RESULTS HANDLING:
When no data is available or results are empty:
- Acknowledge clearly: "I couldn't find any matches for [query] in our current database"
- Suggest alternatives: "You could try searching for related interests like [actual available options]"
- NEVER make up users, events, or data to fill the gap

IMPORTANT GUIDELINES:
- Be conversational and friendly
- If data isn't available for a question, politely explain what the platform can help with
- Always base your responses ONLY on the actual data provided in the API response
- NEVER mention "clusters" or "algorithms" in your responses
- NEVER make up information that isn't in the provided data
- It's better to say "no results found" than to provide false information

Remember that people use this platform to build their social connections, find events, and discover marketplace items.
"""


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_user_profile(user_id):
    """Generate a synthetic user profile with real-world interests and events"""
    # Random age between 18-75
    age = random.randint(18, 75)
    
    # Random location from major cities
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
             "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
             "San Francisco", "Columbus", "Indianapolis", "London", "Tokyo", "Paris", "Berlin"]
    
    location = random.choice(cities)
    
    # Random primary interest category and 1-3 specific interests from that category
    primary_category = random.choice(list(INTEREST_CATEGORIES.keys()))
    interests = random.sample(INTEREST_CATEGORIES[primary_category], 
                             min(random.randint(1, 3), len(INTEREST_CATEGORIES[primary_category])))
    
    # Add 0-2 interests from other random categories
    other_categories = [c for c in INTEREST_CATEGORIES if c != primary_category]
    if other_categories and random.random() < 0.7:  # 70% chance for additional interests
        other_category = random.choice(other_categories)
        other_interests = random.sample(INTEREST_CATEGORIES[other_category], 
                                      min(random.randint(1, 2), len(INTEREST_CATEGORIES[other_category])))
        interests.extend(other_interests)
    
    # Setup preferred events (2-4 events)
    # Higher probability for events related to their interests
    events_pool = []
    # Add events from primary interest category
    if primary_category in EVENTS:
        events_pool.extend(EVENTS[primary_category])
    
    # Add general events
    if 'General' in EVENTS:
        events_pool.extend(EVENTS['General'])
    
    # Add some random events from other categories
    other_event_cats = [c for c in EVENTS if c != primary_category and c != 'General']
    if other_event_cats:
        random_cats = random.sample(other_event_cats, min(2, len(other_event_cats)))
        for cat in random_cats:
            events_pool.extend(EVENTS[cat])
    
    # Select 2-4 unique events
    preferred_events = random.sample(events_pool, min(random.randint(2, 4), len(events_pool)))
    
    # Create marketplace activity (30% chance of having something)
    marketplace_offerings = []
    marketplace_needs = []
    
    if random.random() < 0.3:
        # Items commonly sold/bought in marketplaces
        items = random.sample(ALL_MARKETPLACE_ITEMS, 3)
        
        # 60% chance of offering something
        if random.random() < 0.6:
            marketplace_offerings.append(items[0])
            # 30% chance of offering a second item
            if random.random() < 0.3:
                marketplace_offerings.append(items[1])
        
        # 40% chance of needing something
        if random.random() < 0.4:
            item_needed = items[2] if items[2] not in marketplace_offerings else items[1]
            marketplace_needs.append(item_needed)
    
    return {
        'user_id': user_id,
        'age': age,
        'location': location,
        'interests': interests,
        'preferred_events': preferred_events,
        'marketplace_offerings': marketplace_offerings,
        'marketplace_needs': marketplace_needs,
        'last_active': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_dataset(n_users=2000):
    """Generate a dataset of synthetic user profiles"""
    random.seed(42)  # For reproducibility
    users = [generate_user_profile(i + 1) for i in range(n_users)]
    return pd.DataFrame(users)

class UserClusterer:
    """
    A class to handle the clustering of user data with visualizations
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        self.cluster_labels = None
        self.feature_names = None
        self.cluster_profiles = {}
        self.features_matrix = None
        self.original_df = None
        self.processed_df = None
        self.silhouette_scores = []
        
    def preprocess_data(self, df):
        """
        Preprocess the data for clustering
        """
        # Store original data
        self.original_df = df.copy()
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Create binary features for interests
        for interest in ALL_INTERESTS:
            df_processed[f'interest_{interest}'] = df_processed['interests'].apply(
                lambda x: 1 if interest in x else 0
            )
        
        # Create binary features for event preferences
        for event in ALL_EVENTS:
            df_processed[f'event_{event}'] = df_processed['preferred_events'].apply(
                lambda x: 1 if event in x else 0
            )
        
        # Create binary features for marketplace offerings
        for item in ALL_MARKETPLACE_ITEMS:
            df_processed[f'offering_{item}'] = df_processed['marketplace_offerings'].apply(
                lambda x: 1 if item in x else 0
            )
        
        # Create binary features for marketplace needs
        for item in ALL_MARKETPLACE_ITEMS:
            df_processed[f'need_{item}'] = df_processed['marketplace_needs'].apply(
                lambda x: 1 if item in x else 0
            )
        
        # Extract features for clustering
        numerical_features = ['age']
        categorical_features = ['location']
        interest_features = [f'interest_{i}' for i in ALL_INTERESTS]
        event_features = [f'event_{e}' for e in ALL_EVENTS]
        marketplace_offering_features = [f'offering_{i}' for i in ALL_MARKETPLACE_ITEMS]
        marketplace_need_features = [f'need_{i}' for i in ALL_MARKETPLACE_ITEMS]
        
        
        self.feature_names = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'interest': interest_features,
            'event': event_features,
            'marketplace_offering': marketplace_offering_features,
            'marketplace_need': marketplace_need_features
        }
        
        # Scale numerical features
        X_numerical = df_processed[numerical_features].values
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        
        # Encode categorical features
        X_categorical = df_processed[categorical_features].values
        X_categorical_encoded = self.encoder.fit_transform(X_categorical)
        
        # Get binary features (already 0/1)
        X_interests = df_processed[interest_features].values
        X_events = df_processed[event_features].values
        X_offerings = df_processed[marketplace_offering_features].values
        X_needs = df_processed[marketplace_need_features].values
        
        # Combine all features
        X_combined = np.hstack([
            X_numerical_scaled,
            X_categorical_encoded,
            X_interests,
            X_events,
            X_offerings,
            X_needs
        ])
        
        # Store the feature matrix
        self.features_matrix = X_combined
        
        # Store the processed DataFrame (without the original list columns)
        self.processed_df = df_processed
        
        return X_combined
    
    def fit(self, df):
        """
        Fit the KMeans clustering model to the data
        """
        # Preprocess the data
        X = self.preprocess_data(df)
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, self.cluster_labels)
        
        # Add cluster labels to the processed DataFrame
        self.processed_df['cluster'] = self.cluster_labels
        self.original_df['cluster'] = self.cluster_labels
        
        # Create cluster profiles
        self._create_cluster_profiles()
        
        return self, silhouette_avg
    
    def _create_cluster_profiles(self):
        """Create detailed profiles for each cluster"""
        for cluster_id in range(self.n_clusters):
            cluster_df = self.processed_df[self.processed_df['cluster'] == cluster_id]
            
            # Basic statistics
            profile = {
                'size': len(cluster_df),
                'percentage': len(cluster_df) / len(self.processed_df) * 100,
                'age': {
                    'mean': cluster_df['age'].mean(),
                    'std': cluster_df['age'].std()
                },
                'location_dist': cluster_df['location'].value_counts(normalize=True).to_dict(),
            }
            
            # Top interests
            interest_features = self.feature_names['interest']
            interest_means = cluster_df[interest_features].mean().sort_values(ascending=False)
            profile['top_interests'] = [(col.replace('interest_', ''), val) 
                                      for col, val in interest_means.head(5).items()]
            
            # Top event preferences
            event_features = self.feature_names['event']
            event_means = cluster_df[event_features].mean().sort_values(ascending=False)
            profile['top_events'] = [(col.replace('event_', ''), val) 
                                   for col, val in event_means.head(5).items()]
            
            # Top marketplace offerings
            offering_features = self.feature_names['marketplace_offering']
            offering_means = cluster_df[offering_features].mean().sort_values(ascending=False)
            profile['top_offerings'] = [(col.replace('offering_', ''), val) 
                                      for col, val in offering_means.head(3).items() if val > 0]
            
            # Top marketplace needs
            need_features = self.feature_names['marketplace_need']
            need_means = cluster_df[need_features].mean().sort_values(ascending=False)
            profile['top_needs'] = [(col.replace('need_', ''), val) 
                                   for col, val in need_means.head(3).items() if val > 0]
            
            self.cluster_profiles[cluster_id] = profile
    
    def find_optimal_clusters(self, df, range_n_clusters=range(2, 11)):
        """
        Find the optimal number of clusters using silhouette score
        """
        # Preprocess the data
        X = self.preprocess_data(df)
        
        # Test different numbers of clusters
        self.silhouette_scores = []
        
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, cluster_labels)
            self.silhouette_scores.append((n_clusters, silhouette_avg))
        
        # Find the best number of clusters
        best_n_clusters = max(self.silhouette_scores, key=lambda x: x[1])[0]
        
        # Update the number of clusters
        self.n_clusters = best_n_clusters
        
        return self.silhouette_scores, best_n_clusters
    
    
    def create_visualizations(self):
        """
        Create visualizations and return them as base64 encoded images
        """
        if self.kmeans is None:
            return {}
            
        visualizations = {}
        
        # Apply dimensionality reduction techniques
        features_matrix = self.features_matrix
        
        # 1. PCA for linear dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(features_matrix)
        
        # 2. t-SNE for non-linear dimensionality reduction (better at preserving clusters)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result = tsne.fit_transform(features_matrix)
        
        # Create dataframes for visualization
        viz_df = pd.DataFrame({
            'user_id': self.original_df['user_id'],
            'age': self.original_df['age'],
            'location': self.original_df['location'],
            'cluster': self.cluster_labels,
            'pca_x': pca_result[:, 0],
            'pca_y': pca_result[:, 1],
            'tsne_x': tsne_result[:, 0],
            'tsne_y': tsne_result[:, 1]
        })
        
        # Set up colors for consistent cluster representation
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))
        
        # 1. PCA Visualization
        fig = plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            cluster_data = viz_df[viz_df['cluster'] == i]
            plt.scatter(
                cluster_data['pca_x'], 
                cluster_data['pca_y'],
                c=[colors[i]],
                label=f'Cluster {i} ({len(cluster_data)} users)',
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )
        plt.title('PCA Cluster Visualization', fontsize=14)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')
        visualizations['pca'] = img_data
        plt.close(fig)
        
        # 2. t-SNE Visualization
        fig = plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            cluster_data = viz_df[viz_df['cluster'] == i]
            plt.scatter(
                cluster_data['tsne_x'], 
                cluster_data['tsne_y'],
                c=[colors[i]],
                label=f'Cluster {i} ({len(cluster_data)} users)',
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )
        plt.title('t-SNE Cluster Visualization', fontsize=14)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')
        visualizations['tsne'] = img_data
        plt.close(fig)
        
        # 3. Location Distribution by Cluster
        fig = plt.figure(figsize=(12, 8))
        location_counts = pd.crosstab(
            self.original_df['cluster'], 
            self.original_df['location'],
            normalize='index'
        )
        location_counts.plot(kind='bar', stacked=True, colormap='tab20')
        plt.title('Location Distribution within Each Cluster', fontsize=14)
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Convert to base64
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')
        visualizations['location_dist'] = img_data
        plt.close(fig)
        
        # 4. Silhouette scores visualization
        if self.silhouette_scores:
            fig = plt.figure(figsize=(10, 6))
            x_values = [score[0] for score in self.silhouette_scores]
            y_values = [score[1] for score in self.silhouette_scores]
            plt.plot(x_values, y_values, 'o-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score by Number of Clusters')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_data = base64.b64encode(img_buf.read()).decode('utf-8')
            visualizations['silhouette'] = img_data
            plt.close(fig)
            
        return visualizations
def expand_abbreviations(interest):
    """Expand common abbreviations"""
    abbreviations = {
        'ai': 'artificial intelligence',
        'ml': 'machine learning', 
        'cs': 'computer science',
        'ui': 'user interface',
        'ux': 'user experience',
        'it': 'information technology',
        'vr': 'virtual reality',
        'ar': 'augmented reality',
        'diy': 'do it yourself',
        'cv': 'computer vision'
    }
    
    interest_lower = interest.lower().strip()
    return abbreviations.get(interest_lower, interest)

    
    
  
# Helper Functions for User and Item Matching
def find_users_by_interest(interest, df, fuzzy=True):
    """Find users with a specific interest, with stricter fuzzy matching"""
    expanded_interest = expand_abbreviations(interest)
    normalized_interest = expanded_interest.lower()
    # Remove this line: normalized_interest = interest.lower()
    
    matching_users = []
    
    for _, row in df.iterrows():
        user_interests = [i.lower() for i in row['interests']]
        
        best_match = None
        best_match_score = float('inf')
        
        for user_interest in user_interests:
            # Method 1: Exact match (highest priority)
            if normalized_interest == user_interest:
                best_match = user_interest
                best_match_score = 0
                break
                
            # Method 2: More strict containment check
            if fuzzy:
                # Only allow containment if the lengths are similar (within 50% of each other)
                min_length = min(len(normalized_interest), len(user_interest))
                max_length = max(len(normalized_interest), len(user_interest))
                
                if max_length <= min_length * 1.5:  # Within 50% length difference
                    if normalized_interest in user_interest or user_interest in normalized_interest:
                        length_diff = abs(len(normalized_interest) - len(user_interest))
                        if length_diff < best_match_score:
                            best_match = user_interest
                            best_match_score = length_diff
                
                # Method 3: Levenshtein distance (for typos only)
                if abs(len(normalized_interest) - len(user_interest)) <= 3:
                    distance = levenshtein_distance(normalized_interest, user_interest)
                    max_allowed_distance = min(2, max(1, len(normalized_interest) // 4))
                    
                    if distance <= max_allowed_distance and distance < best_match_score:
                        best_match = user_interest
                        best_match_score = distance
        
        # If we found a match
        if best_match is not None:
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'matched_interest': best_match,
                'match_quality': best_match_score,
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            matching_users.append(user_info)
    
    matching_users.sort(key=lambda x: x['match_quality'])
    return matching_users
def find_users_by_event(event, df, fuzzy=True):
    """Find users interested in a specific event with stricter fuzzy matching"""
    normalized_event = event.lower()
    
    matching_users = []
    
    for _, row in df.iterrows():
        user_events = [e.lower() for e in row['preferred_events']]
        
        best_match = None
        best_match_score = float('inf')
        
        for user_event in user_events:
            # Method 1: Exact match (highest priority)
            if normalized_event == user_event:
                best_match = user_event
                best_match_score = 0
                break
                
            # Method 2: More strict containment check
            if fuzzy:
                # Only allow containment if the lengths are similar (within 50% of each other)
                min_length = min(len(normalized_event), len(user_event))
                max_length = max(len(normalized_event), len(user_event))
                
                if max_length <= min_length * 1.5:  # Within 50% length difference
                    if normalized_event in user_event or user_event in normalized_event:
                        length_diff = abs(len(normalized_event) - len(user_event))
                        if length_diff < best_match_score:
                            best_match = user_event
                            best_match_score = length_diff
                
                # Method 3: Levenshtein distance (for typos only)
                if abs(len(normalized_event) - len(user_event)) <= 3:
                    distance = levenshtein_distance(normalized_event, user_event)
                    max_allowed_distance = min(2, max(1, len(normalized_event) // 4))
                    
                    if distance <= max_allowed_distance and distance < best_match_score:
                        best_match = user_event
                        best_match_score = distance
        
        # If we found a match
        if best_match is not None:
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'matched_event': best_match,
                'match_quality': best_match_score,
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            matching_users.append(user_info)
    
    # Sort results with best matches first
    matching_users.sort(key=lambda x: x['match_quality'])
    return matching_users

def find_users_by_category(category, df):
    """Find users with interests in a specific category"""
    matching_users = []
    
    # Check if the category exists
    if category not in INTEREST_CATEGORIES:
        return matching_users
    
    # Get all interests in the category
    category_interests = INTEREST_CATEGORIES[category]
    
    for _, row in df.iterrows():
        # Check if user has any interests in this category
        if any(interest in category_interests for interest in row['interests']):
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'interests': [i for i in row['interests'] if i in category_interests],
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            
            matching_users.append(user_info)
    
    return matching_users

def find_users_by_location(location, df):
    """Find users in a specific location"""
    # Normalize the location (lowercase for comparison)
    normalized_location = location.lower()
    
    matching_users = []
    
    for _, row in df.iterrows():
        # Check for case-insensitive match
        if row['location'].lower() == normalized_location:
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'interests': row['interests'],
                'preferred_events': row['preferred_events'],
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            
            matching_users.append(user_info)
    
    return matching_users

def find_users_by_age_range(min_age, max_age, df):
    """Find users within a specific age range"""
    matching_users = []
    
    for _, row in df.iterrows():
        if min_age <= row['age'] <= max_age:
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'interests': row['interests'],
                'preferred_events': row['preferred_events'],
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            
            matching_users.append(user_info)
    
    return matching_users

def find_similar_users(user_id, df, limit=5):
    """Find users similar to the given user based on shared interests"""
    # Find the user
    user_data = df[df['user_id'] == user_id]
    
    if user_data.empty:
        return []
    
    # Get the user's cluster if available
    cluster = None
    if 'cluster' in user_data.columns:
        cluster = int(user_data['cluster'].iloc[0])
    
    # Get the user's interests
    user_interests = user_data['interests'].iloc[0]
    
    # Find users with similar interests
    similar_users = []
    
    # If clustering is available, prioritize users in the same cluster
    if cluster is not None:
        candidate_users = df[(df['cluster'] == cluster) & (df['user_id'] != user_id)]
    else:
        candidate_users = df[df['user_id'] != user_id]
    
    for _, row in candidate_users.iterrows():
        other_interests = row['interests']
        
        # Count shared interests
        shared_interests = set(user_interests).intersection(set(other_interests))
        similarity_score = len(shared_interests)
        
        if similarity_score > 0:
            similar_users.append({
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'similarity_score': similarity_score,
                'shared_interests': list(shared_interests)
            })
    
    # Sort by similarity score (number of shared interests)
    similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return similar_users[:limit]

def get_events_by_category(category):
    """Get all events in a specific category"""
    if category in EVENTS:
        return EVENTS[category]
    else:
        return []

def get_interests_by_category(category):
    """Get all interests in a specific category"""
    if category in INTEREST_CATEGORIES:
        return INTEREST_CATEGORIES[category]
    else:
        return []

def suggest_events_for_user(user_id, df):
    """Suggest events for a user based on their interests"""
    # Find the user
    user_data = df[df['user_id'] == user_id]
    
    if user_data.empty:
        return []
    
    # Get the user's interests
    user_interests = user_data['interests'].iloc[0]
    
    # Get the user's current events
    current_events = user_data['preferred_events'].iloc[0]
    
    # Find relevant categories based on interests
    relevant_categories = set()
    for interest in user_interests:
        for category, interests in INTEREST_CATEGORIES.items():
            if interest in interests:
                relevant_categories.add(category)
    
    # Collect potential events from relevant categories
    potential_events = []
    for category in relevant_categories:
        if category in EVENTS:
            potential_events.extend(
                [event for event in EVENTS[category] if event not in current_events]
            )
    
    # Add some general events if available
    if 'General' in EVENTS:
        potential_events.extend(
            [event for event in EVENTS['General'] if event not in current_events]
        )
    
    # Remove duplicates and limit to 10 suggestions
    suggested_events = list(set(potential_events))[:10]
    
    return suggested_events


def send_event_invitations(event_id, event_details, users):
    """Send invitations to users for an event with proper error handling"""
    if not users:
        return 0
    
    try:
        # Ensure creator_id is not None
        creator_id = event_details.get('creator_id')
        if creator_id is None:
            global current_user_id
            creator_id = current_user_id if current_user_id is not None else 0
            event_details['creator_id'] = creator_id
            print(f"Warning: creator_id is None for event {event_id}, setting to {creator_id}")
        
        invitations_sent = 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if using_mongodb:
            # MongoDB version
            batch_messages = []
            
            for user in users:
                # Create personalized invitation
                message = f"You're invited to a {event_details['title']} event on {event_details['date']} at {event_details['time']} at {event_details['location']}. Would you like to attend?"
                
                # Debug print
                print(f"Sending invitation from user {creator_id} to user {user['user_id']}")
                
                # Create message document
                message_doc = {
                    'from_user_id': creator_id,
                    'to_user_id': user['user_id'],
                    'message_type': 'event_invitation',
                    'content': message,
                    'event_id': event_id,
                    'event_details': event_details,
                    'timestamp': timestamp,
                    'is_read': 0,
                    'response': None,
                    'pending_response': False,
                    'conversation_context': {
                        'type': 'event_invitation',
                        'event_id': event_id,
                        'expires_at': (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                
                batch_messages.append(message_doc)
                invitations_sent += 1
            
            # Insert all messages in one batch operation
            if batch_messages:
                db.messages.insert_many(batch_messages)
        else:
            # Original SQLite code
            conn = sqlite3.connect('messages.db')
            c = conn.cursor()
            
            event_json = json.dumps(event_details)
            
            for user in users:
                # Create personalized invitation
                message = f"You're invited to a {event_details['title']} event on {event_details['date']} at {event_details['time']} at {event_details['location']}. Would you like to attend?"
                
                # Debug print
                print(f"Sending invitation from user {creator_id} to user {user['user_id']}")
                
                c.execute('''
                INSERT INTO messages 
                (from_user_id, to_user_id, message_type, content, event_id, event_details, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    creator_id, 
                    user['user_id'], 
                    'event_invitation', 
                    message, 
                    event_id, 
                    event_json, 
                    timestamp
                ))
                
                invitations_sent += 1
            
            conn.commit()
            conn.close()
        
        return invitations_sent
    
    except Exception as e:
        print(f"Error sending invitations: {str(e)}")
        return 0
    
def find_interested_users(event_type, creator_id):
    """Find users who might be interested in an event based on interests or events"""
    # Get the best available dataset
    available_files = os.listdir(app.config['UPLOAD_FOLDER'])
    csv_files = [f for f in available_files if f.endswith('.csv')]
    
    if not csv_files:
        return []
    
    # Prefer clustered files for better matching
    clustered_files = [f for f in csv_files if f.startswith('clustered_')]
    filename = clustered_files[0] if clustered_files else csv_files[0]
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Convert string representations to lists
        if 'interests' in df.columns:
            df['interests'] = df['interests'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        if 'preferred_events' in df.columns:
            df['preferred_events'] = df['preferred_events'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # First, try to match event_type with actual events
        matching_events = []
        event_type_lower = event_type.lower()
        
        # Look through all categories and events to find matches
        for category, events in EVENTS.items():
            for event in events:
                if event.lower() == event_type_lower or event_type_lower in event.lower():
                    matching_events.append(event)
        
        if matching_events:
            print(f"Found matching events: {matching_events}")
            
            # Find users who prefer these events
            event_matching_users = []
            for _, row in df.iterrows():
                if row['user_id'] == creator_id:
                    continue
                
                # Check if user prefers any of the matching events
                if any(event in row['preferred_events'] for event in matching_events):
                    event_matching_users.append({
                        'user_id': int(row['user_id']),
                        'age': int(row['age']),
                        'location': row['location'],
                        'match_reason': 'event preference'
                    })
            
            # If we found users who prefer these events, return them
            if event_matching_users:
                print(f"Found {len(event_matching_users)} users who prefer similar events")
                return event_matching_users
        
        # If no matching events or no users found, fall back to keyword matching
        main_keywords = []
        words = event_type.lower().split()
        for word in words:
            if word not in ['event', 'the', 'a', 'an'] and len(word) > 2:
                main_keywords.append(word)
                
        if not main_keywords and words:
            main_keywords = [words[0]]
        
        print(f"Looking for users with interests matching: {main_keywords}")
        
        # Find matching users based on interests
        interest_matching_users = []
        for _, row in df.iterrows():
            if row['user_id'] == creator_id:
                continue
                
            user_interests_lower = [interest.lower() for interest in row['interests']]
            
            if any(any(keyword in interest for keyword in main_keywords) for interest in user_interests_lower):
                interest_matching_users.append({
                    'user_id': int(row['user_id']),
                    'age': int(row['age']),
                    'location': row['location'],
                    'match_reason': 'interest keyword'
                })
        
        print(f"Found {len(interest_matching_users)} potential users based on interests")
        return interest_matching_users
    
    except Exception as e:
        print(f"Error finding interested users: {str(e)}")
        return []
        
def find_users_by_marketplace_need(item, df):
    """Find users who need a specific item"""
    # Normalize the search item (lowercase for comparison)
    normalized_item = item.lower()
    
    matching_users = []
    
    for _, row in df.iterrows():
        # Check if the user has marketplace needs
        if not isinstance(row['marketplace_needs'], list) or not row['marketplace_needs']:
            continue
            
        # Convert each need to lowercase for comparison
        user_needs = [n.lower() for n in row['marketplace_needs']]
        
        # Check for exact or fuzzy matches
        if any(normalized_item in need or need in normalized_item for need in user_needs):
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'needs': row['marketplace_needs'],
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            
            matching_users.append(user_info)
    
    return matching_users


                
    
def find_users_by_marketplace_offering(item, df):
    """Find users who are offering a specific item"""
    # Normalize the search item (lowercase for comparison)
    normalized_item = item.lower()
    
    matching_users = []
    
    for _, row in df.iterrows():
        # Check if the user has marketplace offerings
        if not isinstance(row['marketplace_offerings'], list) or not row['marketplace_offerings']:
            continue
            
        # Convert each offering to lowercase for comparison
        user_offerings = [o.lower() for o in row['marketplace_offerings']]
        
        # Check for exact or fuzzy matches
        if any(normalized_item in offering or offering in normalized_item for offering in user_offerings):
            user_info = {
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location'],
                'offerings': row['marketplace_offerings'],
                'cluster': int(row['cluster']) if 'cluster' in row else None
            }
            
            matching_users.append(user_info)
    
    return matching_users

# Add these time handling functions from paste-1.txt to paste-2.txt

def time_to_minutes(t):
    """Convert a time string to minutes since midnight"""
    try:
        if ":" in t:
            parts = t.split(":")
            hour = int(parts[0])
            # Handle minute part which might include AM/PM
            minute_part = parts[1]
            if "PM" in minute_part.upper() and hour < 12:
                hour += 12
            elif "AM" in minute_part.upper() and hour == 12:
                hour = 0
            minute = int(''.join(c for c in minute_part if c.isdigit()))
        else:
            # Format like "2PM"
            if "PM" in t.upper() and int(t.rstrip(" APMapm")) < 12:
                hour = int(t.rstrip(" APMapm")) + 12
            elif "AM" in t.upper() and int(t.rstrip(" APMapm")) == 12:
                hour = 0
            else:
                hour = int(t.rstrip(" APMapm"))
            minute = 0
        return hour * 60 + minute
    except Exception as e:
        print(f"Error converting time '{t}' to minutes: {e}")
        # Default to noon if parsing fails
        return 12 * 60

def normalize_time_string(time_str):
    """Normalize time strings for consistent formatting and comparison"""
    try:
        if not time_str or time_str == "NONE":
            return None
        
        # Remove any spaces
        time_str = time_str.strip().upper()
        
        # Handle simple hour formats
        if re.match(r'^\d{1,2}(AM|PM)$', time_str):
            hour = int(time_str.rstrip('AMPM'))
            if 'PM' in time_str and hour < 12:
                hour += 12
            elif 'AM' in time_str and hour == 12:
                hour = 0
            return f"{hour:02d}:00"
        
        # Handle hour:minute formats (might include AM/PM)
        match = re.match(r'^(\d{1,2}):(\d{2})(?:\s*(AM|PM))?$', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3)
            
            if ampm == 'PM' and hour < 12:
                hour += 12
            elif ampm == 'AM' and hour == 12:
                hour = 0
                
            return f"{hour:02d}:{minute:02d}"
        
        # If we can't parse it properly, return original
        return time_str
    except Exception as e:
        print(f"Error normalizing time '{time_str}': {e}")
        return time_str

def calculate_optimal_time(time_data):
    """Calculate optimal meeting time based on participant preferences"""
    try:
        # Get original time
        original_time = "14:00"  # Default to 2:00 PM if not specified
        original_time_minutes = time_to_minutes(original_time)
        
        # Extract times and convert to minutes for analysis
        times_minutes = []
        weights = []
        
        for participant, data in time_data.items():
            status = data.get('status')
            time_str = data.get('time')
            
            if status == 'AVAILABLE':
                # Original event time weight
                times_minutes.append(original_time_minutes)
                weights.append(1.0)  # Full weight for available at original time
            elif status == 'RESCHEDULE' and time_str and time_str != 'NONE':
                times_minutes.append(time_to_minutes(time_str))
                weights.append(0.75)  # Slightly less weight for reschedule requests
        
        # If no time preferences, return default
        if not times_minutes:
            return {
                "best_time": None,
                "time_window": None,
                "attendance_estimate": 0,
                "confidence": "low"
            }
        
        # Return results based on time analysis
        # This is simplified from the original for example purposes
        if len(set(times_minutes)) == 1:
            minute = times_minutes[0]
            best_time = f"{minute//60:02d}:{minute%60:02d}"
            return {
                "best_time": best_time,
                "time_window": f"{best_time} - {best_time}",
                "attendance_estimate": 100,
                "confidence": "medium" 
            }
        
        # Find the most common time
        from collections import Counter
        common_times = Counter(times_minutes)
        most_common_time = common_times.most_common(1)[0][0]
        best_hour = f"{most_common_time//60:02d}:{most_common_time%60:02d}"
        
        return {
            "best_time": best_hour,
            "time_window": f"{best_hour}",
            "attendance_estimate": 75,
            "confidence": "medium"
        }
    except Exception as e:
        print(f"Error calculating optimal time: {e}")
        return {
            "best_time": None,
            "time_window": None,
            "attendance_estimate": 0,
            "confidence": "low"
        }

def analyze_interest_distribution(df):
    """Analyze the distribution of interests across all users"""
    # Count occurrences of each interest
    interest_counts = {}
    
    for _, row in df.iterrows():
        for interest in row['interests']:
            if interest not in interest_counts:
                interest_counts[interest] = 0
            interest_counts[interest] += 1
    
    # Calculate percentage
    total_users = len(df)
    interest_distribution = {
        interest: (count / total_users) * 100
        for interest, count in interest_counts.items()
    }
    
    # Sort by popularity
    sorted_distribution = sorted(
        interest_distribution.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return sorted_distribution

def visualize_interest_distribution(df, top_n=20):
    """Create a visualization of the most popular interests"""
    # Get interest distribution
    interest_dist = analyze_interest_distribution(df)
    
    # Take top N interests
    top_interests = interest_dist[:top_n]
    
    # Extract labels and values
    labels = [item[0] for item in top_interests]
    values = [item[1] for item in top_interests]
    
    # Create horizontal bar chart
    fig = plt.figure(figsize=(12, 8))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel('Percentage of Users (%)')
    plt.ylabel('Interest')
    plt.title(f'Top {top_n} Popular Interests')
    plt.tight_layout()
    
    # Convert to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data

def analyze_event_distribution(df):
    """Analyze the distribution of events across all users"""
    # Count occurrences of each event
    event_counts = {}
    
    for _, row in df.iterrows():
        for event in row['preferred_events']:
            if event not in event_counts:
                event_counts[event] = 0
            event_counts[event] += 1
    
    # Calculate percentage
    total_users = len(df)
    event_distribution = {
        event: (count / total_users) * 100
        for event, count in event_counts.items()
    }
    
    # Sort by popularity
    sorted_distribution = sorted(
        event_distribution.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return sorted_distribution

def visualize_event_distribution(df, top_n=20):
    """Create a visualization of the most popular events"""
    # Get event distribution
    event_dist = analyze_event_distribution(df)
    
    # Take top N events
    top_events = event_dist[:top_n]
    
    # Extract labels and values
    labels = [item[0] for item in top_events]
    values = [item[1] for item in top_events]
    
    # Create horizontal bar chart
    fig = plt.figure(figsize=(12, 8))
    plt.barh(labels, values, color='lightgreen')
    plt.xlabel('Percentage of Users (%)')
    plt.ylabel('Event')
    plt.title(f'Top {top_n} Popular Events')
    plt.tight_layout()
    
    # Convert to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data

def analyze_cluster_by_demographics(clusterer, demographic='age'):
    """Analyze clusters by demographic factors"""
    if clusterer.original_df is None or clusterer.cluster_labels is None:
        return {}
    
    results = {}
    
    if demographic == 'age':
        # Create age groups
        age_bins = [18, 25, 35, 45, 55, 65, 75]
        age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        
        # Add age group column
        df_copy = clusterer.original_df.copy()
        df_copy['age_group'] = pd.cut(df_copy['age'], bins=age_bins, labels=age_labels, right=False)
        
        # Calculate distribution for each cluster
        for cluster_id in range(clusterer.n_clusters):
            cluster_data = df_copy[df_copy['cluster'] == cluster_id]
            age_dist = cluster_data['age_group'].value_counts(normalize=True).to_dict()
            results[cluster_id] = age_dist
    
    elif demographic == 'location':
        # Calculate location distribution for each cluster
        for cluster_id in range(clusterer.n_clusters):
            cluster_data = clusterer.original_df[clusterer.original_df['cluster'] == cluster_id]
            location_dist = cluster_data['location'].value_counts(normalize=True).to_dict()
            results[cluster_id] = location_dist
    
    return results

def visualize_demographic_analysis(clusterer, demographic='age'):
    """Create a visualization of demographic analysis by cluster"""
    analysis = analyze_cluster_by_demographics(clusterer, demographic)
    
    if not analysis:
        return None
    
    # Create a DataFrame from the analysis
    cluster_ids = []
    demographic_values = []
    percentages = []
    
    for cluster_id, dist in analysis.items():
        for value, percentage in dist.items():
            cluster_ids.append(f'Cluster {cluster_id}')
            demographic_values.append(value)
            percentages.append(percentage * 100)  # Convert to percentage
    
    df = pd.DataFrame({
        'Cluster': cluster_ids,
        demographic.capitalize(): demographic_values,
        'Percentage': percentages
    })
    
    # Create a grouped bar chart
    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Cluster', y='Percentage', hue=demographic.capitalize(), data=df)
    plt.title(f'{demographic.capitalize()} Distribution by Cluster', fontsize=14)
    plt.xlabel('Cluster')
    plt.ylabel('Percentage (%)')
    plt.legend(title=demographic.capitalize())
    plt.tight_layout()
    
    # Convert to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data



# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset_route():
    n_users = int(request.form.get('n_users', 2000))
    df = generate_dataset(n_users)
    
    # Save to CSV
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_dataset.csv')
    df.to_csv(csv_path, index=False)
    
    return jsonify({
        'success': True,
        'message': f'Generated dataset with {n_users} users',
        'filename': 'generated_dataset.csv'
    })



@app.route('/perform_clustering', methods=['POST'])
def perform_clustering():
    filename = request.form.get('filename')
    n_clusters = int(request.form.get('n_clusters', 5))
    find_optimal = request.form.get('find_optimal', 'false') == 'true'
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        df = pd.read_csv(filepath)
        
        # Convert string representations of lists to actual lists
        list_columns = ['interests', 'preferred_events', 'marketplace_offerings', 'marketplace_needs']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x))
        
        # Perform clustering
        clusterer = UserClusterer(n_clusters=n_clusters)
        
        if find_optimal:
            silhouette_scores, best_n_clusters = clusterer.find_optimal_clusters(df)
            clusterer, silhouette_avg = clusterer.fit(df)
            
            result = {
                'success': True,
                'message': f'Found optimal number of clusters: {best_n_clusters}',
                'optimal_clusters': best_n_clusters,
                'silhouette_scores': [(int(k), float(v)) for k, v in silhouette_scores],
                'silhouette_avg': float(silhouette_avg)
            }
        else:
            clusterer, silhouette_avg = clusterer.fit(df)
            
            result = {
                'success': True,
                'message': f'Performed clustering with {n_clusters} clusters',
                'silhouette_avg': float(silhouette_avg)
            }
        
        # Generate visualizations
        visualizations = clusterer.create_visualizations()
        
        # Create profile summaries
        profile_summaries = []
        for cluster_id, profile in clusterer.cluster_profiles.items():
            summary = {
                'cluster_id': int(cluster_id),
                'size': int(profile['size']),
                'percentage': float(profile['percentage']),
                'avg_age': float(profile['age']['mean']),
                'top_locations': sorted(profile['location_dist'].items(), key=lambda x: x[1], reverse=True)[:3],
                'top_interests': profile['top_interests'],
                'top_events': profile['top_events']
            }
            
            if 'top_offerings' in profile:
                summary['top_offerings'] = profile['top_offerings']
            
            if 'top_needs' in profile:
                summary['top_needs'] = profile['top_needs']
            
            profile_summaries.append(summary)
        
        # Add to result
        result['visualizations'] = visualizations
        result['cluster_profiles'] = profile_summaries
        
        # Save clustered data back to CSV
        clustered_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'clustered_{filename}')
        clusterer.original_df.to_csv(clustered_filepath, index=False)
        result['clustered_filename'] = f'clustered_{filename}'
        
        # Store clusterer in session (would need to implement proper storage in production)
        # For now, just return the result
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_query', methods=['POST'])
def analyze_query():
    query = request.json.get('query', '')
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    try:
        # Use LLM to classify the query
        response = client.chat.completions.create(
            model="anthropic/claude-3-opus:beta",
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": query}
            ]
        )
        
        classification = json.loads(response.choices[0].message.content)
        
        # Return the classification
        return jsonify({
            'success': True,
            'classification': classification
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def log_session_state(user_id, prefix=""):
    """Log current session state for debugging"""
    try:
        user_specific_key = f"event_creation_state_{user_id}" if user_id else "event_creation_state"
        legacy_data = session.get('event_creation_state', {})
        user_specific_data = session.get(user_specific_key, {})
        
        print(f"{prefix} SESSION DEBUG - user_id: {user_id}")
        print(f"{prefix} SESSION DEBUG - all keys: {list(session.keys())}")
        print(f"{prefix} SESSION DEBUG - legacy data: {legacy_data}")
        print(f"{prefix} SESSION DEBUG - user-specific data: {user_specific_data}")
        
        # Check for mismatch in creator_id
        if (legacy_data and user_specific_data and 
            'event_details' in legacy_data and 'event_details' in user_specific_data and
            legacy_data['event_details'].get('creator_id') != user_specific_data['event_details'].get('creator_id')):
            print(f"{prefix} SESSION DEBUG - WARNING: creator_id mismatch between legacy and user-specific")
    except Exception as e:
        print(f"{prefix} SESSION DEBUG - error logging session: {str(e)}")

# Replace the section in your @app.route('/find_matches', methods=['POST']) function
# where you handle "No results found" with this enhanced version:
class SmartInterestRecommendationAssistant:
    def __init__(self, client, interest_categories):
        self.client = client
        self.interest_categories = interest_categories
        
        # Create a flat list of all available interests
        self.all_available_interests = []
        for interests in interest_categories.values():
            self.all_available_interests.extend(interests)
        self.all_available_interests = list(set(self.all_available_interests))  # Remove duplicates
        self.system_prompt = f"""
You are an AI assistant for Tarp AI that helps users find relevant interests from our community platform.

CRITICAL RULES:
1. You can ONLY recommend interests that exist in our Tarp AI community
2. If no similar interests exist, you MUST say so clearly
3. NEVER suggest interests that are not available on our platform
4. When recommending alternatives, explain the connection/similarity

AVAILABLE INTERESTS ON TARP AI:
{', '.join(self.all_available_interests)}

AVAILABLE CATEGORIES:
{', '.join(interest_categories.keys())}

Your task:
1. Check if the searched interest exists in our community (exact or fuzzy match)
2. If it doesn't exist, find the most contextually similar interests that ARE available
3. If no similar interests exist, clearly state this
4. Always be honest about what's available vs what's not

Response format:
{{
    "interest_found": true/false,
    "exact_matches": ["list of exact matches"],
    "similar_matches": ["list of contextually similar interests that exist"],
    "explanation": "explanation of why these are similar",
    "recommendation_message": "user-friendly message with recommendations or honest 'not available' response"
}}

Examples:

Input: "wrestling"
Response: {{
    "interest_found": false,
    "exact_matches": [],
    "similar_matches": ["Martial Arts", "Weight Training"],
    "explanation": "Wrestling involves combat sports techniques (similar to Martial Arts) and requires significant physical strength training (similar to Weight Training)",
    "recommendation_message": "I couldn't find 'Wrestling' in our Tarp AI community yet, but I found some related interests: Martial Arts (combat techniques) and Weight Training (physical conditioning). Would you like to search for people with these interests instead?"
}}

Input: "machine learning" 
Response: {{
    "interest_found": false,
    "exact_matches": [],
    "similar_matches": ["AI", "Programming"],
    "explanation": "Machine learning is a subset of AI and typically involves programming skills",
    "recommendation_message": "I couldn't find 'Machine Learning' specifically in our community, but we have related interests: AI and Programming. These are closely related to machine learning work. Would you like to search for people interested in these instead?"
}}

Input: "underwater basket weaving"
Response: {{
    "interest_found": false,
    "exact_matches": [],
    "similar_matches": [],
    "explanation": "No contextually similar interests found in our community",
    "recommendation_message": "I couldn't find 'Underwater Basket Weaving' or any closely related interests in our Tarp AI community yet. You might want to browse our available categories like Arts, Crafts, or Sports to see what interests are available."
}}
"""
        


    def get_smart_recommendations(self, searched_interest, search_results=None):
        """
        Get smart recommendations for interests that aren't found in the database
        """
        try:
            # First check if we actually found any results
            has_results = False
            if search_results:
                for key, value in search_results.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, list) and len(sub_value) > 0:
                                has_results = True
                                break
                    elif isinstance(value, list) and len(value) > 0:
                        has_results = True
                        break
                    if has_results:
                        break
            
            # If we found results, no need for recommendations
            if has_results:
                return None
            
            # Build context for the LLM
            context = f"""
SEARCHED INTEREST: "{searched_interest}"
SEARCH RESULTS: No matches found in database

Find the most contextually similar interests from the available database that relate to "{searched_interest}".
Only recommend interests that actually exist in the provided list.
"""

            # Call the LLM
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            print(f"Smart Recommendation LLM Response: {response_text}")
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "interest_found": False,
                    "exact_matches": [],
                    "similar_matches": [],
                    "explanation": "Could not analyze recommendations",
                    "recommendation_message": f"I couldn't find '{searched_interest}' in our database. You might want to browse our available interest categories to see what's available."
                }
            
            return result
            
        except Exception as e:
            print(f"Error in smart recommendations: {str(e)}")
            return {
                "interest_found": False,
                "exact_matches": [],
                "similar_matches": [],
                "explanation": f"Error occurred: {str(e)}",
                "recommendation_message": f"I couldn't find '{searched_interest}' in our database. Please try browsing our available categories."
            }

    def verify_interests_exist(self, interests_list):
        """
        Verify that a list of interests actually exist in the database
        """
        verified_interests = []
        for interest in interests_list:
            # Check exact match (case insensitive)
            for available_interest in self.all_available_interests:
                if interest.lower() == available_interest.lower():
                    verified_interests.append(available_interest)
                    break
        
        return verified_interests

    def get_available_categories_info(self):
        """
        Get information about available categories and their interests
        """
        categories_info = {}
        for category, interests in self.interest_categories.items():
            categories_info[category] = {
                'count': len(interests),
                'sample_interests': interests[:5]  # Show first 5 as examples
            }
        
        return categories_info


# Initialize the smart recommendation assistant
smart_recommender = SmartInterestRecommendationAssistant(client, INTEREST_CATEGORIES)

class SmartEventRecommendationAssistant:
    def __init__(self, client, events_dict):
        self.client = client
        self.events_dict = events_dict
        
        # Create a flat list of all available events
        self.all_available_events = []
        for events in events_dict.values():
            self.all_available_events.extend(events)
        self.all_available_events = list(set(self.all_available_events))
        
        self.system_prompt = f"""
You are an AI assistant for Tarp AI that helps users find relevant events from our community platform.

CRITICAL RULES:
1. You can ONLY recommend events that exist in our Tarp AI community
2. If no similar events exist, you MUST say so clearly
3. NEVER suggest events that are not available on our platform
4. When recommending alternatives, explain the connection/similarity

AVAILABLE EVENTS ON TARP AI:
{', '.join(self.all_available_events)}

AVAILABLE CATEGORIES:
{', '.join(events_dict.keys())}

Your task:
1. Check if the searched event exists in our community (exact or fuzzy match)
2. If it doesn't exist, find the most contextually similar events that ARE available
3. If no similar events exist, clearly state this
4. Always be honest about what's available vs what's not

Response format:
{{
    "event_found": true/false,
    "exact_matches": ["list of exact matches"],
    "similar_matches": ["list of contextually similar events that exist"],
    "explanation": "explanation of why these are similar",
    "recommendation_message": "user-friendly message with recommendations or honest 'not available' response"
}}

Examples:

Input: "python workshop"
Response: {{
    "event_found": false,
    "exact_matches": [],
    "similar_matches": ["Programming Workshops", "Tech Meetups"],
    "explanation": "Python is a programming language, so Programming Workshops would cover similar content. Tech Meetups often include programming discussions and networking.",
    "recommendation_message": "I couldn't find 'Python Workshop' specifically in our Tarp AI community, but we have Programming Workshops (which would cover Python and other languages) and Tech Meetups (great for programming discussions). Would you like to search for people attending these instead?"
}}

Input: "underwater basket weaving class"
Response: {{
    "event_found": false,
    "exact_matches": [],
    "similar_matches": [],
    "explanation": "No contextually similar events found in our community",
    "recommendation_message": "I couldn't find 'Underwater Basket Weaving Class' or any closely related events in our Tarp AI community yet. You might want to browse our available categories like Arts, Crafts, or General events to see what's available."
}}
"""


    def get_smart_event_recommendations(self, searched_event, search_results=None):
        """Get smart recommendations for events that aren't found in the database"""
        try:
            # First check if we actually found any results
            has_results = False
            if search_results and search_results.get('matching_users'):
                has_results = len(search_results['matching_users']) > 0
            
            # If we found results, no need for recommendations
            if has_results:
                return None
            
            # Build context for the LLM
            context = f"""
SEARCHED EVENT: "{searched_event}"
SEARCH RESULTS: No matches found in database

Find the most contextually similar events from the available database that relate to "{searched_event}".
Only recommend events that actually exist in the provided list.
"""

            # Call the LLM
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            print(f"Smart Event Recommendation LLM Response: {response_text}")
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "event_found": False,
                    "exact_matches": [],
                    "similar_matches": [],
                    "explanation": "Could not analyze recommendations",
                    "recommendation_message": f"I couldn't find '{searched_event}' in our database. You might want to browse our available event categories to see what's available."
                }
            
            return result
            
        except Exception as e:
            print(f"Error in smart event recommendations: {str(e)}")
            return {
                "event_found": False,
                "exact_matches": [],
                "similar_matches": [],
                "explanation": f"Error occurred: {str(e)}",
                "recommendation_message": f"I couldn't find '{searched_event}' in our database. Please try browsing our available categories."
            }
# Initialize the smart event recommendation assistant
smart_event_recommender = SmartEventRecommendationAssistant(client, EVENTS)
@app.route('/find_matches', methods=['POST'])
def find_matches():
    classification = request.json.get('classification', {})
    filename = request.json.get('filename', '')
    
    if not classification or not filename:
        return jsonify({'success': False, 'error': 'Missing classification or filename'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        df = pd.read_csv(filepath)
        
        # Convert string representations of lists to actual lists
        list_columns = ['interests', 'preferred_events', 'marketplace_offerings', 'marketplace_needs']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x))
        
        query_type = classification.get('query_type', '')
        entities = classification.get('entities', {})
        
        results = {}
        
        # Your existing query processing logic here...
        if query_type == 'interest_search':
            interests = entities.get('interests', [])
            
            if interests:
                # Find users for each interest
                interest_results = {}
                for interest in interests:
                    matching_users = find_users_by_interest(interest, df)
                    interest_results[interest] = matching_users
                
                # If multiple interests, find users who match all interests
                if len(interests) > 1:
                    common_users = []
                    all_user_ids = set()
                    
                    for users in interest_results.values():
                        user_ids = {user['user_id'] for user in users}
                        all_user_ids = all_user_ids.union(user_ids) if all_user_ids else user_ids
                    
                    for user_id in all_user_ids:
                        matched_interests = []
                        
                        for interest, users in interest_results.items():
                            if user_id in {user['user_id'] for user in users}:
                                matched_interests.append(interest)
                        
                        if matched_interests:
                            user_data = df[df['user_id'] == user_id].iloc[0]
                            common_users.append({
                                'user_id': int(user_id),
                                'age': int(user_data['age']),
                                'location': user_data['location'],
                                'matched_interests': matched_interests,
                                'total_matches': len(matched_interests),
                                'cluster': int(user_data['cluster']) if 'cluster' in user_data else None
                            })
                    
                    # Sort by number of matching interests
                    common_users.sort(key=lambda x: x['total_matches'], reverse=True)
                    interest_results['common_users'] = common_users
                
                results['interest_results'] = interest_results
        
        # ... (your other query type handling code) ...
        
        # ENHANCED: Check if we have any results with smart recommendations
        has_results = False
        if results:
            for key, value in results.items():
                if isinstance(value, dict):
                    # Check nested dictionaries for lists with data
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list) and len(sub_value) > 0:
                            has_results = True
                            break
                    if has_results:
                        break
                elif isinstance(value, list) and len(value) > 0:
                    has_results = True
                    break
        
        # Generate response using LLM with smart recommendations
        if not has_results:
            user_message = request.json.get('original_query', '')
            
            # ENHANCED: Use smart recommendations for interest searches
            if query_type == 'interest_search':
                interests = entities.get('interests', [])
                if interests:
                    searched_interest = interests[0]  # Take the first interest
                    
                    # Get smart recommendations using the new assistant
                    recommendations = smart_recommender.get_smart_recommendations(searched_interest, results)
                    
                    if recommendations and recommendations.get('similar_matches'):
                        # Verify that recommended interests actually exist in our database
                        verified_interests = smart_recommender.verify_interests_exist(recommendations['similar_matches'])
                        
                        if verified_interests:
                            response_prompt = f"""
User searched for: {user_message}

SMART RECOMMENDATION ANALYSIS:
- Searched interest: "{searched_interest}" 
- Found in database: {recommendations.get('interest_found', False)}
- Similar interests available: {', '.join(verified_interests)}
- Explanation: {recommendations.get('explanation', '')}

Create a helpful response that:
1. Clearly states that "{searched_interest}" is not in our database
2. Explains the similar interests we DO have: {', '.join(verified_interests)}
3. Explains why these are related: {recommendations.get('explanation', '')}
4. Asks if they want to search for these alternatives instead

Be honest about what's not available, but helpful about what IS available.
"""
                        else:
                            response_prompt = f"""
User searched for: {user_message}

ANALYSIS: "{searched_interest}" is not in our database and no closely related interests are available.

Create a response that:
1. Clearly states that "{searched_interest}" is not available
2. Suggests browsing our available categories: {', '.join(INTEREST_CATEGORIES.keys())}
3. Be honest that we don't have this specific interest or close alternatives

Do NOT suggest interests that don't exist in our database.
"""
                    else:
                        response_prompt = f"""
User searched for: {user_message}

ANALYSIS: "{searched_interest}" is not in our database and no similar interests found.

Available categories: {', '.join(INTEREST_CATEGORIES.keys())}

Create a response that:
1. Honestly states we don't have this interest
2. Suggests browsing available categories
3. Do NOT suggest specific interests that might not exist

Be helpful but honest about limitations.
"""
                else:
                    # Fallback for malformed interest searches
                    response_prompt = f"""
User searched for: {user_message}

No specific interest detected or no results found.

Available categories: {', '.join(INTEREST_CATEGORIES.keys())}

Respond helpfully but only suggest browsing categories, not specific interests.
"""
            else:
                # For non-interest searches, use existing logic
                response_prompt = f"""
User searched for: {user_message}

No results found in the database for this query.

Available categories: {', '.join(INTEREST_CATEGORIES.keys())}

Respond helpfully about what the platform can help with, but be honest about limitations.
"""
        else:
            # We have results - use the original prompt
            response_prompt = f"""
User query: {request.json.get('original_query', '')}

API response data:
{json.dumps(results, indent=2)}

Based on this data, generate a helpful and friendly response. Focus on highlighting the most relevant matches and explaining why they might be interesting to the user.
"""

        llm_response = client.chat.completions.create(
            model="anthropic/claude-3-opus:beta",
            messages=[
                {"role": "system", "content": RESPONSE_GENERATOR_PROMPT},
                {"role": "user", "content": response_prompt}
            ]
        )
        
        # Prepare final response with summary statistics (existing code)
        summary_stats = {}
        
        if 'interest_results' in results:
            # Count total unique users across all interests
            all_user_ids = set()
            for interest, users in results['interest_results'].items():
                if interest != 'common_users':
                    all_user_ids.update({user['user_id'] for user in users})
            
            summary_stats['interest_search'] = {
                'total_interests': len(results['interest_results']) - (1 if 'common_users' in results['interest_results'] else 0),
                'total_unique_users': len(all_user_ids)
            }
            
            # Add stats for each interest
            for interest, users in results['interest_results'].items():
                if interest != 'common_users':
                    summary_stats['interest_search'][interest] = len(users)
            
            # Add stats for common users if available
            if 'common_users' in results['interest_results']:
                summary_stats['interest_search']['users_with_all_interests'] = len(results['interest_results']['common_users'])
        
        # ... (rest of your existing summary stats code) ...
        
        # Return both raw results, generated response, and summary statistics
        return jsonify({
            'success': True,
            'results': results,
            'response': llm_response.choices[0].message.content,
            'summary_stats': summary_stats,
            'smart_recommendations': recommendations if not has_results and query_type == 'interest_search' else None
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})
# Additional routes for enhanced functionality

@app.route('/recommend_connections', methods=['POST'])
def recommend_connections():
    user_id = request.json.get('user_id')
    filename = request.json.get('filename', '')
    
    if not user_id or not filename:
        return jsonify({'success': False, 'error': 'Missing user ID or filename'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        df = pd.read_csv(filepath)
        
        # Convert string representations of lists to actual lists
        list_columns = ['interests', 'preferred_events', 'marketplace_offerings', 'marketplace_needs']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x))
        
        # Find the user
        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            return jsonify({'success': False, 'error': 'User not found'})
        
        user_data = user_data.iloc[0]
        
        # Find similar users based on interests
        similar_users = find_similar_users(user_id, df, limit=10)
        
        # Find users from same location
        location_users = find_users_by_location(user_data['location'], df)
        location_users = [user for user in location_users if user['user_id'] != user_id][:5]
        
        # Find users in the same cluster
        if 'cluster' in user_data and user_data['cluster'] is not None:
            cluster_users = df[
                (df['cluster'] == user_data['cluster']) & 
                (df['user_id'] != user_id)
            ].sample(n=min(5, len(df[(df['cluster'] == user_data['cluster']) & (df['user_id'] != user_id)])))
            
            cluster_users_data = []
            for _, row in cluster_users.iterrows():
                shared_interests = set(row['interests']).intersection(set(user_data['interests']))
                cluster_users_data.append({
                    'user_id': int(row['user_id']),
                    'age': int(row['age']),
                    'location': row['location'],
                    'interests': row['interests'],
                    'shared_interests': list(shared_interests)
                })
        else:
            cluster_users_data = []
        
        # Marketplace connections (users offering what this user needs)
        marketplace_matches = []
        
        if ('marketplace_needs' in user_data and 
            isinstance(user_data['marketplace_needs'], list) and 
            user_data['marketplace_needs']):
            
            for need in user_data['marketplace_needs']:
                offering_users = find_users_by_marketplace_offering(need, df)
                offering_users = [user for user in offering_users if user['user_id'] != user_id]
                
                for user in offering_users:
                    user['matching_item'] = need
                    user['match_type'] = 'offering_what_you_need'
                
                marketplace_matches.extend(offering_users[:3])  # Top 3 matches for each need
        
        # Users needing what this user offers
        if ('marketplace_offerings' in user_data and 
            isinstance(user_data['marketplace_offerings'], list) and 
            user_data['marketplace_offerings']):
            
            for offering in user_data['marketplace_offerings']:
                needing_users = find_users_by_marketplace_need(offering, df)
                needing_users = [user for user in needing_users if user['user_id'] != user_id]
                
                for user in needing_users:
                    user['matching_item'] = offering
                    user['match_type'] = 'needing_what_you_offer'
                
                marketplace_matches.extend(needing_users[:3])  # Top 3 matches for each offering
        
        # Event-based connections (users attending same events)
        event_connections = []
        
        if user_data['preferred_events']:
            for event in user_data['preferred_events']:
                event_users = find_users_by_event(event, df)
                event_users = [user for user in event_users if user['user_id'] != user_id]
                
                for user in event_users[:3]:  # Top 3 matches for each event
                    user['matching_event'] = event
                    event_connections.append(user)
        
        # Combine all recommendations and remove duplicates
        all_recommendations = []
        seen_user_ids = set()
        
        # Add interest-based connections (highest priority)
        for user in similar_users:
            if user['user_id'] not in seen_user_ids:
                user['connection_type'] = 'interest'
                all_recommendations.append(user)
                seen_user_ids.add(user['user_id'])
        
        # Add event-based connections
        for user in event_connections:
            if user['user_id'] not in seen_user_ids:
                user['connection_type'] = 'event'
                all_recommendations.append(user)
                seen_user_ids.add(user['user_id'])
        
        # Add marketplace connections
        for user in marketplace_matches:
            if user['user_id'] not in seen_user_ids:
                user['connection_type'] = 'marketplace'
                all_recommendations.append(user)
                seen_user_ids.add(user['user_id'])
        
        # Add location-based connections
        for user in location_users:
            if user['user_id'] not in seen_user_ids:
                user['connection_type'] = 'location'
                all_recommendations.append(user)
                seen_user_ids.add(user['user_id'])
        
        # Add cluster-based connections
        for user in cluster_users_data:
            if user['user_id'] not in seen_user_ids:
                user['connection_type'] = 'cluster'
                all_recommendations.append(user)
                seen_user_ids.add(user['user_id'])
        
        # Limit to 15 total recommendations
        all_recommendations = all_recommendations[:15]
        
        # Generate response using LLM
        response_prompt = f"""
User ID: {user_id}
User location: {user_data['location']}
User age: {user_data['age']}
User interests: {', '.join(user_data['interests'])}
User preferred events: {', '.join(user_data['preferred_events'])}

I've found {len(all_recommendations)} potential connections for this user. Here are some highlights:
- {len([u for u in all_recommendations if u['connection_type'] == 'interest'])} people with shared interests
- {len([u for u in all_recommendations if u['connection_type'] == 'event'])} people attending the same events
- {len([u for u in all_recommendations if u['connection_type'] == 'marketplace'])} marketplace connections
- {len([u for u in all_recommendations if u['connection_type'] == 'location'])} people in the same location

Based on this data, generate a friendly and conversational recommendation of potential connections for this user. Highlight why each connection might be valuable and what they have in common.
"""

        llm_response = client.chat.completions.create(
            model="anthropic/claude-3-opus:beta",
            messages=[
                {"role": "system", "content": RESPONSE_GENERATOR_PROMPT},
                {"role": "user", "content": response_prompt}
            ]
        )
        
        return jsonify({
            'success': True,
            'user_info': {
                'user_id': int(user_id),
                'age': int(user_data['age']),
                'location': user_data['location'],
                'interests': user_data['interests'],
                'preferred_events': user_data['preferred_events']
            },
            'recommendations': all_recommendations,
            'recommendation_types': {
                'interest_based': len([u for u in all_recommendations if u['connection_type'] == 'interest']),
                'event_based': len([u for u in all_recommendations if u['connection_type'] == 'event']),
                'marketplace': len([u for u in all_recommendations if u['connection_type'] == 'marketplace']),
                'location_based': len([u for u in all_recommendations if u['connection_type'] == 'location']),
                'cluster_based': len([u for u in all_recommendations if u['connection_type'] == 'cluster'])
            },
            'response': llm_response.choices[0].message.content
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/event_recommendations', methods=['POST'])
def event_recommendations():
    user_id = request.json.get('user_id')
    filename = request.json.get('filename', '')
    
    if not user_id or not filename:
        return jsonify({'success': False, 'error': 'Missing user ID or filename'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        df = pd.read_csv(filepath)
        
        # Convert string representations of lists to actual lists
        list_columns = ['interests', 'preferred_events', 'marketplace_offerings', 'marketplace_needs']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x))
        
        # Find the user
        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            return jsonify({'success': False, 'error': 'User not found'})
        
        user_data = user_data.iloc[0]
        
        # Get suggested events
        suggested_events = suggest_events_for_user(user_id, df)
        
        # Get event details
        event_details = []
        
        for event in suggested_events:
            # Find users attending this event
            event_users = find_users_by_event(event, df)
            event_users = [user for user in event_users if user['user_id'] != user_id]
            
            # Find which category this event belongs to
            event_category = None
            for category, events in EVENTS.items():
                if event in events:
                    event_category = category
                    break
            
            # Find if any similar users (based on interests) are attending
            similar_users = find_similar_users(user_id, df, limit=20)
            similar_user_ids = {user['user_id'] for user in similar_users}
            
            similar_attendees = [
                user for user in event_users 
                if user['user_id'] in similar_user_ids
            ]
            
            event_details.append({
                'event_name': event,
                'category': event_category,
                'total_attendees': len(event_users),
                'similar_attendees': len(similar_attendees),
                'sample_attendees': event_users[:5]  # Sample 5 attendees
            })
        
        # Generate response using LLM
        response_prompt = f"""
User ID: {user_id}
User interests: {', '.join(user_data['interests'])}
User current events: {', '.join(user_data['preferred_events'])}

Based on this user's profile, I've recommended these events:
{json.dumps(event_details, indent=2)}

Generate a friendly, conversational recommendation of events for this user. For each event, explain why it might be interesting to them based on their interests and who they might meet there. Highlight events where their similar connections will be attending.
"""

        llm_response = client.chat.completions.create(
            model="anthropic/claude-3-opus:beta",
            messages=[
                {"role": "system", "content": RESPONSE_GENERATOR_PROMPT},
                {"role": "user", "content": response_prompt}
            ]
        )
        
        return jsonify({
            'success': True,
            'user_info': {
                'user_id': int(user_id),
                'interests': user_data['interests'],
                'current_events': user_data['preferred_events']
            },
            'recommended_events': event_details,
            'response': llm_response.choices[0].message.content
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# 1. Add Levenshtein distance calculation for fuzzy matching
def levenshtein_distance(s1, s2):
    """Calculate the edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # If s2 is empty, the distance is just the length of s1
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            # Get the minimum to append to the current row
            current_row.append(min(insertions, deletions, substitutions))
        
        # Update previous row
        previous_row = current_row
    
    return previous_row[-1]

# Complete Smart Event Creation Assistant Class with the missing handler method

class SmartEventCreationAssistant:
    def __init__(self, client, events_dict, valid_cities):
        self.client = client
        self.events_dict = events_dict
        self.valid_cities = valid_cities
        
        # Get lists for reference
        self.all_events = []
        for events_list in events_dict.values():
            self.all_events.extend(events_list)
        
        # System prompt for the Event Creation LLM
        self.system_prompt = f"""
You are Tarp's Event Creation Assistant - a smart assistant that helps users create events step by step.

AVAILABLE DATA:
- Valid Event Types: {', '.join(self.all_events)}
- Valid Locations: {', '.join(valid_cities)}

REQUIRED FIELDS FOR EVENT CREATION:
1. Event Type (title) - MUST be from valid event types
2. Day - When the event happens
3. Time - What time the event starts  
4. Location - MUST be from valid locations

YOUR JOB:
- Analyze what information the user has provided
- Identify what's missing from the 4 required fields
- Ask for the NEXT missing field in logical order: event type → day → time → location
- Validate provided information against available data
- Once ALL 4 fields are complete, ask for confirmation

RESPONSE FORMAT:
Always respond with JSON:
{{
    "message": "Your natural response to the user",
    "action": "ask_event_type|ask_date|ask_time|ask_location|confirm|validation_error|complete",
    "event_details": {{"title": "", "date": "", "time": "", "location": ""}},
    "missing_fields": ["list of what's still needed"],
    "validation_issues": ["any problems with provided data"],
    "next_step": "what to ask for next"
}}

VALIDATION RULES:
- Event type MUST match one from the valid list (use fuzzy matching)
- Location MUST match one from the valid cities (use fuzzy matching)
- If invalid event/location provided, set action to "validation_error"

CONVERSATION FLOW EXAMPLES:

User: "I want to create an event"
→ Missing: event type, date, time, location
→ Ask for event type first

User: "create a tech meetup"
→ Has: event type (validate "tech meetup" → "Tech Meetups")
→ Missing: date, time, location
→ Ask for date next

User: "create a travel meetup on Saturday at 2pm"
→ Has: event type, date, time
→ Missing: location
→ Ask for location

User: "create a cooking class on Friday at 6pm in Dallas"
→ Has: all 4 fields
→ Validate and ask for confirmation

User: "create a jumbo event"
→ Invalid event type
→ validation_error with suggestions

EXAMPLES:

Input: "I want to create an event"
Response: {{
    "message": "Great! I'd love to help you create an event! 🎉 What type of event would you like to organize? We have options like Tech Meetups, Cooking Classes, Art Classes, Travel Meetups, and many more!",
    "action": "ask_event_type",
    "event_details": {{"title": "", "date": "", "time": "", "location": ""}},
    "missing_fields": ["title", "date", "time", "location"],
    "next_step": "event_type"
}}

Input: "tech meetup on Saturday"
Response: {{
    "message": "Awesome! A Tech Meetup on Saturday sounds great! 💻 What time would you like it to start?",
    "action": "ask_time", 
    "event_details": {{"title": "Tech Meetups", "date": "Saturday", "time": "", "location": ""}},
    "missing_fields": ["time", "location"],
    "next_step": "time"
}}

Input: "jumbo event"
Response: {{
    "message": "I'd love to help, but I don't see 'jumbo event' in our available event types. 😅 How about one of these popular options: Tech Meetups, Cooking Classes, Art Classes, or Travel Meetups? Which one interests you?",
    "action": "validation_error",
    "event_details": {{"title": "", "date": "", "time": "", "location": ""}},
    "validation_issues": ["Invalid event type: jumbo event"],
    "next_step": "event_type"
}}

REMEMBER: Always ask for only ONE missing field at a time, in order: event type → date → time → location → confirmation
"""

    # In your SmartEventCreationAssistant.process_user_input method, 
# replace the JSON parsing section with this safer version:

    def process_user_input(self, user_message, current_event_details=None, current_step=""):
        """
        Process user input through the smart LLM assistant
        """
        try:
            # Build context for the LLM
            context = f"""
    CURRENT STEP: {current_step or "starting new event"}
    CURRENT EVENT DETAILS: {current_event_details or {}}
    USER MESSAGE: "{user_message}"

    Analyze what the user provided, validate it, determine what's missing, and respond with the next step.
    """

            # Call the LLM
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            print(f"Event Creation LLM Response: {response_text}")
            
            # ENHANCED JSON PARSING with error handling
            try:
                # First, try direct parsing
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                
                # Try to clean the response text
                cleaned_text = response_text
                
                # Remove control characters
                import re
                cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
                
                # Try to extract JSON from response if wrapped in other text
                json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        print("Failed to parse cleaned JSON, using fallback")
                        result = None
                else:
                    result = None
                
                # Ultimate fallback if JSON parsing completely fails
                if result is None:
                    print("Using fallback response due to JSON parsing failure")
                    result = {
                        "message": "I'm having trouble parsing the response. Let's continue with your event creation. What type of event would you like to organize?",
                        "action": "ask_event_type",
                        "event_details": current_event_details or {"title": "", "date": "", "time": "", "location": ""},
                        "missing_fields": ["title", "date", "time", "location"],
                        "validation_issues": [f"JSON parsing error: {str(e)}"],
                        "next_step": "event_type"
                    }
            
            # Additional validation and fuzzy matching
            event_details = result.get("event_details", {})
            
            # Validate and correct event type
            if event_details.get("title"):
                corrected_title = self._find_best_event_match(event_details["title"])
                if corrected_title:
                    event_details["title"] = corrected_title
                    result["event_details"] = event_details
                else:
                    # Invalid event type
                    result["action"] = "validation_error"
                    result["validation_issues"] = [f"'{event_details['title']}' is not a valid event type"]
            
            # Validate and correct location
            if event_details.get("location"):
                corrected_location = self._find_best_location_match(event_details["location"])
                if corrected_location:
                    event_details["location"] = corrected_location
                    result["event_details"] = event_details
                else:
                    # Invalid location
                    result["action"] = "validation_error"
                    result["validation_issues"] = [f"'{event_details['location']}' is not a valid location"]
            
            return result
            
        except Exception as e:
            print(f"Error in Event Creation LLM: {str(e)}")
            return {
                "message": "I'm having a small hiccup. Let's start over - what type of event would you like to create?",
                "action": "ask_event_type", 
                "event_details": {"title": "", "date": "", "time": "", "location": ""},
                "missing_fields": ["title", "date", "time", "location"],
                "validation_issues": [f"Processing error: {str(e)}"],
                "next_step": "event_type"
            }
        
    def _find_best_event_match(self, event_input):
        """Find best matching event using fuzzy logic"""
        event_lower = event_input.lower()
        
        # Exact match
        for event in self.all_events:
            if event_lower == event.lower():
                return event
        
        # Partial match
        for event in self.all_events:
            if event_lower in event.lower() or event.lower() in event_lower:
                return event
        
        return None
    
    def _find_best_location_match(self, location_input):
        """Find best matching location using fuzzy logic"""
        location_lower = location_input.lower()
        
        # Exact match
        for city in self.valid_cities:
            if location_lower == city.lower():
                return city
        
        # Partial match
        for city in self.valid_cities:
            if location_lower in city.lower() or city.lower() in location_lower:
                return city
        
        return None

    # THIS IS THE MISSING METHOD - THE "FINAL CALL" YOU WERE ASKING ABOUT
    
    def handle_smart_event_creation(self, query_type, user_message, entities, event_creation_state, session_key, user_id):
        try:
            if query_type == 'event_creation':
                print("Smart Event Creation Assistant activated")
                
                # Get current event details and step
                current_details = event_creation_state.get('event_details', {"title": "", "date": "", "time": "", "location": ""}) if event_creation_state else {"title": "", "date": "", "time": "", "location": ""}
                current_step = event_creation_state.get('step', '') if event_creation_state else ''
                
                print(f"Current step: {current_step}")
                print(f"Current details: {current_details}")
                
                # Handle confirmation step specially
                if current_step in ['confirm', 'confirm_details']:
                    if any(word in user_message.lower() for word in ['yes', 'yeah', 'sure', 'ok', 'confirm']):
                        # User confirmed - create the event
                        event_id = f"event_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id}"
                        current_details['event_id'] = event_id
                        current_details['creator_id'] = user_id
                        
                        # Save event
                        save_event(current_details)
                        
                        # Find and invite users
                        matching_users = find_interested_users(current_details['title'], user_id)
                        invitations_sent = send_event_invitations(event_id, current_details, matching_users)
                        
                        # Set 15-minute waiting state
                        waiting_key = f"waiting_for_responses_{user_id}"
                        deadline = datetime.now() + timedelta(minutes=15)
                        
                        from flask import session
                        session[waiting_key] = {
                            'event_id': event_id,
                            'deadline': deadline.strftime("%Y-%m-%d %H:%M:%S"),
                            'invitations_sent': invitations_sent,
                            'event_details': current_details,
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Clear event creation session
                        session.pop(session_key, None)
                        session.modified = True
                        
                        confirmation_message = f"""🎉 **Fantastic! Your '{current_details['title']}' event has been created!**

    📅 **Event Details:**
    📆 **Date:** {current_details['date']}
    ⏰ **Time:** {current_details['time']}
    📍 **Location:** {current_details['location']}

    ✉️ **Invitations Sent:** {invitations_sent} interested users have been invited!

    ⏳ **What happens next?**
    • I'm now collecting responses from invited users
    • **Please wait 15 minutes** for the response period to complete
    • You'll automatically receive analytics when responses are ready
    • After 15 minutes, you can filter and manage your attendees

    ⏰ **Response Period:** 15 minutes remaining
    💡 **Tip:** I'll notify you automatically when responses are collected - no need to check back!"""
                        
                        return {
                            'success': True,
                            'query': user_message,
                            'classification': {"query_type": "event_creation", "entities": entities},
                            'results': {
                                'event_details': current_details, 
                                'invitations_sent': invitations_sent,
                                'waiting_period_started': True,
                                'deadline': deadline.strftime("%Y-%m-%d %H:%M:%S")
                            },
                            'answer': confirmation_message
                        }
                    else:
                        # User cancelled
                        from flask import session
                        session.pop(session_key, None)
                        session.modified = True
                        
                        return {
                            'success': True,
                            'query': user_message,
                            'classification': {"query_type": "event_creation", "entities": entities},
                            'results': {},
                            'answer': "No problem! I've cancelled the event creation. Feel free to start over anytime! 😊"
                        }
                
                # Regular event creation flow
                # Use the smart assistant to process the input
                result = self.process_user_input(user_message, current_details, current_step)
                
                # Extract the response
                action = result.get('action', 'ask_event_type')
                message = result.get('message', 'Let me help you create an event!')
                event_details = result.get('event_details', {})
                next_step = result.get('next_step', 'event_type')
                
                # Ensure creator_id is set
                if 'creator_id' not in event_details:
                    event_details['creator_id'] = user_id
                
                # Update session state
                from flask import session
                if action == 'complete':
                    # All fields collected, ask for confirmation
                    session[session_key] = {
                        'step': 'confirm',
                        'event_details': event_details
                    }
                    
                    # Add confirmation message
                    confirmation_prompt = f"""Perfect! Here's your event summary:

    📅 **Event:** {event_details.get('title', 'Event')}
    📆 **Date:** {event_details.get('date', 'TBD')}
    ⏰ **Time:** {event_details.get('time', 'TBD')}
    📍 **Location:** {event_details.get('location', 'TBD')}

    Would you like me to create this event and send invitations? (Say 'yes' to confirm or 'no' to cancel)"""
                    
                    message = confirmation_prompt
                    
                else:
                    # Still collecting information
                    session[session_key] = {
                        'step': next_step,
                        'event_details': event_details
                    }
                
                session.modified = True
                
                # IMPORTANT: Return the result for regular event creation flow
                return {
                    'success': True,
                    'query': user_message,
                    'classification': {"query_type": "event_creation", "entities": entities},
                    'results': {'creation_state': result},
                    'answer': message
                }
            
            # If not event creation query type, return error
            else:
                return {
                    'success': False,
                    'query': user_message,
                    'classification': {"query_type": query_type, "entities": entities},
                    'results': {},
                    'answer': f"This method only handles event creation, but received query type: {query_type}"
                }
        
        except Exception as e:
            print(f"Error in handle_smart_event_creation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return error result instead of None
            return {
                'success': False,
                'query': user_message,
                'classification': {"query_type": query_type, "entities": entities},
                'results': {},
                'answer': f"Sorry, there was an error creating your event: {str(e)}"
            }
smart_event_assistant = SmartEventCreationAssistant(client, EVENTS, VALID_CITIES) # Smart Event Scheduling Assistant
class SmartEventSchedulingAssistant:
    def __init__(self, client):
        self.client = client
        
        # System prompt for the Event Scheduling LLM
        self.system_prompt = """
You are Tarp's Event Scheduling Assistant - a smart assistant that processes responses to event invitations.

YOUR JOB:
Analyze user responses to event invitations and determine their availability/scheduling preferences.

RESPONSE CATEGORIES:
1. AVAILABLE - User can attend at the original time
2. UNAVAILABLE - User cannot attend at all  
3. RESCHEDULE - User wants a different time
4. UNCLEAR - Cannot determine user's intent

USER RESPONSE PATTERNS:

AVAILABLE responses:
- "yes", "yeah", "yep", "sure", "ok", "okay"
- "I can attend", "I'll be there", "count me in"
- "sounds good", "works for me", "perfect"
- "👍", "✅", affirmative emojis

UNAVAILABLE responses:  
- "no", "nope", "can't make it", "won't be able to attend"
- "I'm busy", "I have other plans", "not available"
- "sorry, can't attend", "I'll pass"
- "❌", "👎", negative emojis

RESCHEDULE responses:
- "can we do [time] instead?", "how about [time]?"
- "I prefer [time]", "[time] would work better"
- "can we move it to [time]?", "what about [time]?"
- Any message containing a different time suggestion

UNCLEAR responses:
- Ambiguous messages that don't clearly indicate availability
- Questions about the event details
- Messages that don't relate to attendance

TIME EXTRACTION:
When user suggests a reschedule, extract the time they mention:
- "6pm", "6:00 PM", "18:00"
- "3 o'clock", "three pm" 
- "evening", "afternoon", "morning" (convert to approximate times)
- "later", "earlier" (relative times)

RESPONSE FORMAT:
Always respond with JSON:
{
    "message": "Your natural response acknowledging their preference",
    "availability": "AVAILABLE|UNAVAILABLE|RESCHEDULE|UNCLEAR", 
    "time": "extracted time or NONE",
    "confidence": "high|medium|low",
    "reasoning": "brief explanation of your decision"
}

EXAMPLES:

User: "yes"
Response: {
    "message": "Great! I've marked you as available for the event. Looking forward to seeing you there! 🎉",
    "availability": "AVAILABLE",
    "time": "NONE", 
    "confidence": "high",
    "reasoning": "Clear affirmative response"
}

User: "can we do 6pm instead?"
Response: {
    "message": "I understand you'd prefer 6pm instead. I've noted your preference for rescheduling to 6pm. ⏰",
    "availability": "RESCHEDULE", 
    "time": "6pm",
    "confidence": "high",
    "reasoning": "Clear reschedule request with specific time"
}

User: "I can't make it"
Response: {
    "message": "No problem! I've marked you as unavailable for this event. Thanks for letting us know! 👍",
    "availability": "UNAVAILABLE",
    "time": "NONE",
    "confidence": "high", 
    "reasoning": "Clear unavailability statement"
}

User: "what's the event about?"
Response: {
    "message": "I see you have questions about the event. Would you like me to provide more details, or are you indicating whether you can attend?",
    "availability": "UNCLEAR",
    "time": "NONE",
    "confidence": "low",
    "reasoning": "Question about event rather than availability response"
}

IMPORTANT: Be natural and conversational while being precise about availability classification.
"""

    def process_invitation_response(self, user_message, event_details):
        """
        Process a user's response to an event invitation
        """
        try:
            # Build context for the LLM
            context = f"""
EVENT INVITATION DETAILS:
- Event: {event_details.get('title', 'Event')}
- Date: {event_details.get('date', 'TBD')}
- Time: {event_details.get('time', 'TBD')}
- Location: {event_details.get('location', 'TBD')}

USER RESPONSE: "{user_message}"

Analyze this response to the event invitation and determine the user's availability/preferences.
"""

            # Call the LLM
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3  # Lower temperature for consistent classification
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            print(f"Event Scheduling LLM Response: {response_text}")
            
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Extract JSON from response if wrapped in other text
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Fallback if JSON parsing fails
                    result = {
                        "message": "Thank you for your response! I'll make note of it.",
                        "availability": "UNCLEAR",
                        "time": "NONE",
                        "confidence": "low",
                        "reasoning": "Could not parse response"
                    }
            
            # Normalize time format if provided
            if result.get("time") and result["time"] != "NONE":
                result["time"] = self._normalize_time(result["time"])
            
            return result
            
        except Exception as e:
            print(f"Error in Event Scheduling LLM: {str(e)}")
            return {
                "message": "I received your response about the event. Thank you!",
                "availability": "UNCLEAR",
                "time": "NONE", 
                "confidence": "low",
                "reasoning": f"Processing error: {str(e)}"
            }
    
    def _normalize_time(self, time_str):
        """Normalize time strings to consistent format"""
        try:
            time_str = time_str.strip().lower()
            
            # Handle common time formats
            if re.match(r'^\d{1,2}(am|pm)$', time_str):
                return time_str
            elif re.match(r'^\d{1,2}:\d{2}(am|pm)?$', time_str):
                return time_str
            elif 'evening' in time_str:
                return '7pm'
            elif 'afternoon' in time_str:
                return '3pm'
            elif 'morning' in time_str:
                return '10am'
            elif 'later' in time_str:
                return 'later'
            elif 'earlier' in time_str:
                return 'earlier'
            else:
                return time_str
                
        except Exception:
            return time_str

# Initialize the smart scheduling assistant
    def handle_smart_event_scheduling(self, query_type, user_message, entities, user_id, request):
            """Handle event scheduling using the smart LLM assistant"""
            
            if query_type == 'event_scheduling':
                print("Smart Event Scheduling Assistant activated")
                
                # Create a session key for event scheduling
                scheduling_session_key = f"event_scheduling_{user_id}" if user_id else "event_scheduling"
                
                # Check if event_message_id was provided in the request
                event_message_id = request.json.get('event_message_id') if hasattr(request, 'json') else None
                
                # Get the event ID and details (using your existing logic)
                event_id = None
                event_details = None
                
                if event_message_id:
                    print(f"Using provided event_message_id: {event_message_id}")
                    # Get event ID from specific message (your existing code)
                    if using_mongodb:
                        message_doc = db.messages.find_one({'_id': ObjectId(event_message_id)})
                        if message_doc:
                            event_id = message_doc.get('event_id')
                            event_details_raw = message_doc.get('event_details')
                            if isinstance(event_details_raw, dict):
                                event_details = event_details_raw
                            else:
                                event_details = json.loads(event_details_raw) if event_details_raw else {}
                    else:
                        # SQLite version (your existing code)
                        conn = sqlite3.connect('messages.db')
                        c = conn.cursor()
                        c.execute('SELECT event_id, event_details FROM messages WHERE id = ?', (event_message_id,))
                        result = c.fetchone()
                        conn.close()
                        
                        if result:
                            event_id = result[0]
                            event_details = json.loads(result[1]) if result[1] else {}
                else:
                    print("No event_message_id provided, searching for recent invitation...")
                    # AUTO-DETECT: Find the most recent unresponded event invitation (your existing code)
                    if using_mongodb:
                        recent_invitation = db.messages.find_one({
                            'to_user_id': user_id,
                            'message_type': {'$in': ['event_invitation', 'reschedule_confirmation']},  # ← NEW LINE
                            'response': {'$exists': False}
                        }, sort=[('timestamp', -1)])
                        
                        if recent_invitation:
                            event_id = recent_invitation.get('event_id')
                            event_details_raw = recent_invitation.get('event_details')
                            event_message_id = str(recent_invitation['_id'])
                            
                            if isinstance(event_details_raw, dict):
                                event_details = event_details_raw
                            else:
                                event_details = json.loads(event_details_raw) if event_details_raw else {}
                    else:
                        # SQLite version (your existing code)
                        conn = sqlite3.connect('messages.db')
                        c = conn.cursor()
                        c.execute('''
                            SELECT id, event_id, event_details FROM messages 
                            WHERE to_user_id = ? AND message_type = 'event_invitation' AND response IS NULL
                            ORDER BY timestamp DESC LIMIT 1
                        ''', (user_id,))
                        result = c.fetchone()
                        conn.close()
                        
                        if result:
                            event_message_id = result[0]
                            event_id = result[1]
                            event_details = json.loads(result[2]) if result[2] else {}
                
                # If no event found
                if not event_id or not event_details:
                    return {
                        'success': True,
                        'query': user_message,
                        'classification': {"query_type": "event_scheduling", "entities": entities},
                        'results': {},
                        'answer': "I'd like to help with scheduling, but I'm not sure which event you're referring to. Could you please specify which event you're discussing?"
                    }
                
                # Process the response with smart assistant
                result = self.process_event_response_smart(user_message, event_details)
                
                # Update the database with this response
                save_event_response(
                    event_id,
                    user_id,
                    result.get('availability'),
                    result.get('time'),
                    user_message
                )
                
                # Mark the message as read and add response
                if event_message_id:
                    if using_mongodb:
                        db.messages.update_one(
                            {'_id': ObjectId(event_message_id)},
                            {'$set': {'is_read': 1, 'response': user_message}}
                        )
                    else:
                        conn = sqlite3.connect('messages.db')
                        c = conn.cursor()
                        c.execute('UPDATE messages SET is_read = 1, response = ? WHERE id = ?', 
                                (user_message, event_message_id))
                        conn.commit()
                        conn.close()
                
                return {
                    'success': True,
                    'query': user_message,
                    'classification': {"query_type": "event_scheduling", "entities": entities},
                    'results': {'scheduling_result': result},
                    'answer': result.get('message', "Thank you for your response to the event invitation.")
    
            }
    def process_event_response_smart(self, user_message, event_details):
        try:
            result = self.process_invitation_response(user_message, event_details)
        
                # Extract results
            availability = result.get('availability', 'UNCLEAR')
            time = result.get('time', 'NONE')
            message = result.get('message', 'Thank you for your response!')
            confidence = result.get('confidence', 'medium')
            reasoning = result.get('reasoning', '')
            
            print(f"Scheduling result: {availability}, time: {time}, confidence: {confidence}")
            print(f"Reasoning: {reasoning}")
            
            # Update event details if rescheduling
            updated_event_details = event_details.copy()
            if availability == 'RESCHEDULE' and time != 'NONE':
                updated_event_details['has_user_rescheduled'] = True
                updated_event_details['last_mentioned_time'] = time
                updated_event_details['current_time'] = time
            
            return {
                'event_details': updated_event_details,
                'availability': availability,
                'time': time,
                'message': message,
                'confidence': confidence,
                'status': 'processed'
            }
            
        except Exception as e:
            print(f"Error processing event response: {str(e)}")
            return {
                'event_details': event_details,
                'message': "Thank you for your response to the event invitation.",
                'availability': 'UNCLEAR',
                'time': 'NONE',
                'confidence': 'low',
                'status': 'error'
            }
# Initialize the smart scheduling assistant
smart_scheduling_assistant = SmartEventSchedulingAssistant(client)

# Replace your entire EventDeadlineChecker class with this complete version:

class EventDeadlineChecker:
    def __init__(self):
        self.running = False
        self.check_interval = 30  # Check every 30 seconds
        
    def start(self):
        """Start the background deadline checker"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, daemon=True)
        self.thread.start()
        print("Event deadline checker started")
        
    def stop(self):
        """Stop the background deadline checker"""
        self.running = False
        
    def _check_loop(self):
        """Main loop that checks for expired deadlines"""
        while self.running:
            try:
                self._check_expired_deadlines()
            except Exception as e:
                print(f"Error in deadline checker: {str(e)}")
            
            time.sleep(self.check_interval)
    
    def _check_expired_deadlines(self):
        """Check for events with expired 15-minute deadlines"""
        current_time = datetime.now()
        print(f"[DEADLINE CHECK] Running at {current_time}")
        
        # IMPORTANT: Check session-based waiting states
        # Since Flask sessions are per-user, we need to check the database for expired events
        
        if using_mongodb:
            # Find events created in the last 20 minutes that might need deadline checking
            cutoff_time = (current_time - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
            
            recent_events = db.events.find({
                'created_at': {'$gte': cutoff_time},
                'deadline_notification_sent': {'$ne': True}  # Haven't sent notification yet
            })
            
            for event in recent_events:
                event_id = event['event_id']
                creator_id = event['creator_id']
                created_at = datetime.strptime(event['created_at'], "%Y-%m-%d %H:%M:%S")
                deadline = created_at + timedelta(minutes=15)
                
                print(f"[DEADLINE CHECK] Checking event {event_id}, deadline: {deadline}, current: {current_time}")
                
                # Check if deadline passed
                if current_time > deadline:
                    print(f"[DEADLINE CHECK] Event {event_id} deadline expired! Sending analytics...")
                    
                    # Send analytics notification and mark as sent
                    self._send_analytics_notification(event_id, creator_id)
                    
                    # Mark that we've sent the notification
                    db.events.update_one(
                        {'event_id': event_id},
                        {'$set': {'deadline_notification_sent': True}}
                    )
        else:
            # SQLite version
            conn_events = sqlite3.connect('social_platform.db')
            c_events = conn_events.cursor()
            
            # Add the missing column if it doesn't exist
            try:
                c_events.execute('ALTER TABLE events ADD COLUMN deadline_notification_sent INTEGER DEFAULT 0')
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Find recent events that haven't had notifications sent
            cutoff_time = (current_time - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
            
            c_events.execute('''
                SELECT event_id, creator_id, created_at FROM events 
                WHERE created_at >= ? AND (deadline_notification_sent IS NULL OR deadline_notification_sent = 0)
            ''', (cutoff_time,))
            
            events = c_events.fetchall()
            
            for event_id, creator_id, created_at_str in events:
                created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                deadline = created_at + timedelta(minutes=15)
                
                print(f"[DEADLINE CHECK] Checking event {event_id}, deadline: {deadline}, current: {current_time}")
                
                if current_time > deadline:
                    print(f"[DEADLINE CHECK] Event {event_id} deadline expired! Sending analytics...")
                    
                    # Send analytics notification
                    self._send_analytics_notification(event_id, creator_id)
                    
                    # Mark as sent
                    c_events.execute('UPDATE events SET deadline_notification_sent = 1 WHERE event_id = ?', (event_id,))
            
            conn_events.commit()
            conn_events.close()
    def _send_analytics_notification(self, event_id, creator_id):

        try:
            print(f"[ENHANCED ANALYTICS] Generating analytics for event {event_id}, creator {creator_id}")
            
            # Use the enhanced analytics function
            enhanced_result = generate_enhanced_analytics_with_rescheduling(event_id, creator_id)
            
            if not enhanced_result:
                print(f"[ENHANCED ANALYTICS] Could not generate enhanced analytics for event {event_id}")
                # Fallback to basic analytics
                analytics = get_event_response_analytics(event_id)
                if not analytics:
                    return
                    
                summary = f"""🎉 **Response Collection Complete!**
        Your event received {analytics['total_responded']} out of {analytics['total_invited']} responses ({analytics['response_rate']}% response rate):
        ✅ **Available**: {analytics['available_count']} people
        ❌ **Unavailable**: {analytics['unavailable_count']} people  
        ⏰ **Want to reschedule**: {analytics['reschedule_count']} people"""
            else:
                analytics = enhanced_result['enhanced_analytics']
                summary = enhanced_result['summary_message']
            
            # Send enhanced analytics message to creator
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if using_mongodb:
                analytics_message = {
                    'from_user_id': 0,  # System message
                    'to_user_id': creator_id,
                    'message_type': 'enhanced_event_analytics',
                    'content': summary,
                    'event_id': event_id,
                    'event_details': None,
                    'timestamp': timestamp,
                    'is_read': 0,
                    'response': None,
                    'analytics_data': analytics
                }
                
                # Add rescheduling analysis if available
                if enhanced_result and 'rescheduling_analysis' in enhanced_result:
                    analytics_message['rescheduling_analysis'] = enhanced_result['rescheduling_analysis']
                
                db.messages.insert_one(analytics_message)
            else:
                # SQLite version
                conn = sqlite3.connect('messages.db')
                c = conn.cursor()
                
                c.execute('''
                INSERT INTO messages 
                (from_user_id, to_user_id, message_type, content, event_id, timestamp, is_read)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (0, creator_id, 'enhanced_event_analytics', summary, event_id, timestamp, 0))
                
                conn.commit()
                conn.close()
            
            print(f"✅ Enhanced analytics notification sent to user {creator_id} for event {event_id}")
            
        except Exception as e:
            print(f"Error sending enhanced analytics notification: {str(e)}")
            import traceback
            traceback.print_exc()
    # Replace your EventDeadlineChecker._check_expired_deadlines method with this:
deadline_checker = EventDeadlineChecker()


class IntelligentReschedulingAssistant:
    def __init__(self, client, db, using_mongodb):
        self.client = client
        self.db = db
        self.using_mongodb = using_mongodb
        
        self.system_prompt = """
You are Tarp's Intelligent Rescheduling Assistant. Your job is to help event creators make smart decisions about event timing based on participant responses.

RESCHEDULING DECISION SCENARIOS:

SCENARIO 1: More people want to reschedule than are available
- Ask creator: "Should we reschedule to the popular time preference?"
- If YES: Start reschedule flow
- If NO: Stick to original time

SCENARIO 2: Equal or fewer reschedule requests than available
- Ask creator: "Most people can make the original time. Proceed as planned?"
- If YES: Confirm with available people
- If NO: Still allow manual reschedule option

RESPONSE FORMAT:
Always respond with JSON:
{
    "message": "Your conversational message to the creator",
    "action": "ask_reschedule_decision|confirm_original_time|start_reschedule_flow|proceed_as_planned",
    "recommendation": "your recommendation based on the data",
    "popular_time": "most requested time if applicable",
    "reasoning": "explain why you're making this recommendation"
}

EXAMPLES:

Analytics showing 8 reschedule requests vs 5 available:
{
    "message": "📊 **Interesting situation!** 8 people want to reschedule (mostly to 6pm) while only 5 can make the original time. Should we reschedule to 6pm to accommodate more people?",
    "action": "ask_reschedule_decision", 
    "recommendation": "reschedule",
    "popular_time": "6pm",
    "reasoning": "More people prefer the new time, could increase attendance"
}

Analytics showing 3 reschedule requests vs 10 available:
{
    "message": "📊 **Great response!** 10 people can make the original time and only 3 want to reschedule. Should we proceed with the original time as planned?",
    "action": "confirm_original_time",
    "recommendation": "keep_original", 
    "popular_time": null,
    "reasoning": "Majority prefer original time, better to stick with it"
}
"""

    def analyze_rescheduling_situation(self, analytics):
        """Analyze analytics and determine what to recommend to the event creator"""
        try:
            available_count = analytics['available_count']
            reschedule_count = analytics['reschedule_count']
            reschedule_times = analytics.get('reschedule_times', {})
            
            # Find the most popular reschedule time
            popular_time = None
            popular_time_count = 0
            
            if reschedule_times:
                popular_time = max(reschedule_times.items(), key=lambda x: x[1])
                popular_time_count = popular_time[1]
                popular_time = popular_time[0]
            
            # Build context for the LLM
            context = f"""
ANALYTICS SUMMARY:
- Available for original time: {available_count} people
- Want to reschedule: {reschedule_count} people
- Most popular reschedule time: {popular_time} ({popular_time_count} people)
- Other reschedule preferences: {reschedule_times}

Determine what recommendation to make to the event creator based on this data.
"""

            # Call the LLM for intelligent decision
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            print(f"Rescheduling LLM Response: {response_text}")
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                if reschedule_count > available_count:
                    result = {
                        "message": f"📊 {reschedule_count} people want to reschedule while {available_count} can make the original time. Should we consider rescheduling?",
                        "action": "ask_reschedule_decision",
                        "recommendation": "reschedule",
                        "popular_time": popular_time,
                        "reasoning": "More people want to reschedule than are available"
                    }
                else:
                    result = {
                        "message": f"📊 {available_count} people can make the original time and {reschedule_count} want to reschedule. Should we proceed as planned?",
                        "action": "confirm_original_time", 
                        "recommendation": "keep_original",
                        "popular_time": None,
                        "reasoning": "Majority prefer original time"
                    }
            
            return result
            
        except Exception as e:
            print(f"Error in rescheduling analysis: {str(e)}")
            return {
                "message": "I've analyzed the responses. Would you like to proceed with the original time or consider rescheduling?",
                "action": "ask_reschedule_decision",
                "recommendation": "manual_decision",
                "popular_time": None,
                "reasoning": f"Analysis error: {str(e)}"
            }

    def handle_creator_rescheduling_decision(self, creator_response, event_id, analytics):

        try:

            creator_response_lower = creator_response.lower().strip()
            
            # Get the original recommendation context
            available_count = analytics['available_count']
            reschedule_count = analytics['reschedule_count']
            
            # Determine what the system recommended
            system_recommended_reschedule = reschedule_count > available_count
            
            print(f"🎯 DECISION CONTEXT:")
            print(f"   Available: {available_count}, Reschedule requests: {reschedule_count}")
            print(f"   System recommended: {'RESCHEDULE' if system_recommended_reschedule else 'KEEP ORIGINAL'}")
            print(f"   User response: '{creator_response}'")
            
            # Parse creator's decision based on context
            affirmative_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'proceed', 'go ahead']
            negative_words = ['no', 'nope', 'cancel', 'stop']
            explicit_reschedule_words = ['reschedule', 'change', 'new time', 'different time']
            explicit_keep_words = ['keep', 'original', 'stick', 'stay', 'same time']
            
            # Check for explicit intentions first
            if any(word in creator_response_lower for word in explicit_reschedule_words):
                decision = 'reschedule'
                print("🔄 EXPLICIT: User explicitly wants to reschedule")
                
            elif any(word in creator_response_lower for word in explicit_keep_words):
                decision = 'keep_original'  
                print("✅ EXPLICIT: User explicitly wants to keep original")
                
            # Handle "yes/no" based on what system recommended
            elif any(word in creator_response_lower for word in affirmative_words):
                if system_recommended_reschedule:
                    decision = 'reschedule'
                    print("🔄 CONTEXTUAL: User said 'yes' to reschedule recommendation")
                else:
                    decision = 'keep_original'
                    print("✅ CONTEXTUAL: User said 'yes' to keep original recommendation")
                    
            elif any(word in creator_response_lower for word in negative_words):
                if system_recommended_reschedule:
                    decision = 'keep_original'
                    print("✅ CONTEXTUAL: User said 'no' to reschedule recommendation")
                else:
                    decision = 'reschedule'
                    print("🔄 CONTEXTUAL: User said 'no' to keep original recommendation")
                    
            else:
                decision = 'unclear'
                print("❓ UNCLEAR: Could not determine user's intention")
            
            print(f"🎯 FINAL DECISION: {decision}")
            
            # Execute the decision
            if decision == 'reschedule':
                return self._start_reschedule_flow(event_id, analytics)
            elif decision == 'keep_original':
                return self._confirm_original_time_flow(event_id, analytics)
            else:
                return {
                    'success': True,
                    'answer': "I'm not sure what you decided. Please say:\n• 'yes' to follow my recommendation\n• 'reschedule' to change the time\n• 'keep original' to stick with the original time"
                }
                
        except Exception as e:
            print(f"Error handling creator decision: {str(e)}")
            return {
            'success': False,
            'answer': f"Error processing your decision: {str(e)}"
        }
    def _start_reschedule_flow(self, event_id, analytics):
        """Start the rescheduling flow - ask available people if they want to join the new time"""
        try:
            # Get the most popular reschedule time
            reschedule_times = analytics.get('reschedule_times', {})
            if not reschedule_times:
                return {
                    'success': False,
                    'answer': "No reschedule preferences found. Cannot proceed with rescheduling."
                }
            
            popular_time = max(reschedule_times.items(), key=lambda x: x[1])
            new_time = popular_time[0]
            support_count = popular_time[1]
            
            # Get event details
            event_details = get_event_details(event_id)
            if not event_details:
                return {
                    'success': False,
                    'answer': "Could not retrieve event details for rescheduling."
                }
            
            # Get available attendees (those who said yes to original time)
            available_attendees = self._get_available_attendees(event_id)
            
            # Create reschedule confirmation message for available attendees
            reschedule_message = f"""🔄 **Event Time Change Proposal**

Your event **{event_details['title']}** was originally scheduled for {event_details['time']}, but {support_count} people have requested to move it to **{new_time}**.

The event creator has decided to consider rescheduling to accommodate more people.

**Would you be available at the new time ({new_time}) instead?**

📅 **Event:** {event_details['title']}
📍 **Location:** {event_details['location']} (unchanged)
🗓️ **Date:** {event_details['date']} (unchanged)
⏰ **NEW TIME:** {new_time}

Please respond with 'yes' if you can make the new time, or 'no' if you prefer the original time."""

            # Send reschedule confirmation to available attendees
            messages_sent = 0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for attendee in available_attendees:
                try:
                    if self.using_mongodb:
                        reschedule_message_doc = {
                            'from_user_id': event_details['creator_id'],
                            'to_user_id': attendee['user_id'],
                            'message_type': 'reschedule_confirmation',
                            'content': reschedule_message,
                            'event_id': event_id,
                            'event_details': event_details,
                            'new_time': new_time,
                            'timestamp': timestamp,
                            'is_read': 0,
                            'response': None
                        }
                        self.db.messages.insert_one(reschedule_message_doc)
                    else:
                        # SQLite version
                        conn = sqlite3.connect('messages.db')
                        c = conn.cursor()
                        c.execute('''
                        INSERT INTO messages 
                        (from_user_id, to_user_id, message_type, content, event_id, event_details, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            event_details['creator_id'], 
                            attendee['user_id'], 
                            'reschedule_confirmation', 
                            reschedule_message, 
                            event_id, 
                            json.dumps(event_details), 
                            timestamp
                        ))
                        conn.commit()
                        conn.close()
                    
                    messages_sent += 1
                    
                except Exception as e:
                    print(f"Error sending reschedule message to user {attendee['user_id']}: {e}")
            
            # Update event creator's state to track reschedule process
            self._set_creator_state(event_details['creator_id'], 'reschedule_confirmation', {
                'event_id': event_id,
                'new_time': new_time,
                'available_attendees_messaged': messages_sent,
                'original_reschedule_count': analytics['reschedule_count'],
                'started_at': timestamp
            })
            
            return {
                'success': True,
                'answer': f"""✅ **Rescheduling Process Started!**

I've sent reschedule confirmation messages to all {messages_sent} people who were available for the original time.

📊 **Current Status:**
• **New proposed time:** {new_time}
• **People who requested this time:** {support_count}
• **Available people being asked:** {messages_sent}

⏳ **Next Steps:**
• I'm collecting responses from those who were originally available
• Once they respond, I'll combine them with the {support_count} people who wanted {new_time}
• You'll get a final attendee list for the rescheduled event

**The rescheduling process is now in motion! 🚀**"""
            }
            
        except Exception as e:
            print(f"Error in reschedule flow: {str(e)}")
            return {
                'success': False,
                'answer': f"Error starting reschedule process: {str(e)}"
            }

    def _confirm_original_time_flow(self, event_id, analytics):
        """Confirm with available people and notify reschedule-requesters that time stays the same"""
        try:
            # Get event details
            event_details = get_event_details(event_id)
            if not event_details:
                return {
                    'success': False,
                    'answer': "Could not retrieve event details."
                }
            
            # Get available attendees
            available_attendees = self._get_available_attendees(event_id)
            
            # Get reschedule requesters
            reschedule_requesters = self._get_reschedule_requesters(event_id)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Message 1: Confirmation to available attendees
            confirmation_message = f"""✅ **Event Confirmed!**

Great news! Your event **{event_details['title']}** is confirmed for the original time.

📅 **Final Event Details:**
🎯 **Event:** {event_details['title']}
📅 **Date:** {event_details['date']}
⏰ **Time:** {event_details['time']}
📍 **Location:** {event_details['location']}

👥 **Confirmed Attendees:** {len(available_attendees)} people

**See you there! 🎉**"""

            # Message 2: Notice to reschedule requesters
            reschedule_notice = f"""📅 **Event Time Update**

Thank you for your response to **{event_details['title']}**.

The event creator has decided to keep the original time due to majority preference.

📅 **Final Event Details:**
🎯 **Event:** {event_details['title']}
📅 **Date:** {event_details['date']}
⏰ **Time:** {event_details['time']} (original time)
📍 **Location:** {event_details['location']}

If you can make it at this time, you're still welcome to join! Otherwise, we hope to see you at future events. 😊"""

            # Send confirmation messages to available attendees
            confirmations_sent = 0
            for attendee in available_attendees:
                try:
                    if self.using_mongodb:
                        confirmation_doc = {
                            'from_user_id': event_details['creator_id'],
                            'to_user_id': attendee['user_id'],
                            'message_type': 'event_confirmation',
                            'content': confirmation_message,
                            'event_id': event_id,
                            'event_details': event_details,
                            'timestamp': timestamp,
                            'is_read': 0
                        }
                        self.db.messages.insert_one(confirmation_doc)
                    else:
                        conn = sqlite3.connect('messages.db')
                        c = conn.cursor()
                        c.execute('''
                        INSERT INTO messages 
                        (from_user_id, to_user_id, message_type, content, event_id, event_details, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            event_details['creator_id'], 
                            attendee['user_id'], 
                            'event_confirmation', 
                            confirmation_message, 
                            event_id, 
                            json.dumps(event_details), 
                            timestamp
                        ))
                        conn.commit()
                        conn.close()
                    
                    confirmations_sent += 1
                    
                except Exception as e:
                    print(f"Error sending confirmation to user {attendee['user_id']}: {e}")

            # Send notices to reschedule requesters
            notices_sent = 0
            for requester in reschedule_requesters:
                try:
                    if self.using_mongodb:
                        notice_doc = {
                            'from_user_id': event_details['creator_id'],
                            'to_user_id': requester['user_id'],
                            'message_type': 'reschedule_notice',
                            'content': reschedule_notice,
                            'event_id': event_id,
                            'event_details': event_details,
                            'timestamp': timestamp,
                            'is_read': 0
                        }
                        self.db.messages.insert_one(notice_doc)
                    else:
                        conn = sqlite3.connect('messages.db')
                        c = conn.cursor()
                        c.execute('''
                        INSERT INTO messages 
                        (from_user_id, to_user_id, message_type, content, event_id, event_details, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            event_details['creator_id'], 
                            requester['user_id'], 
                            'reschedule_notice', 
                            reschedule_notice, 
                            event_id, 
                            json.dumps(event_details), 
                            timestamp
                        ))
                        conn.commit()
                        conn.close()
                    
                    notices_sent += 1
                    
                except Exception as e:
                    print(f"Error sending notice to user {requester['user_id']}: {e}")

            # Clear creator's state as process is complete
            self._clear_creator_state(event_details['creator_id'])

            return {
                'success': True,
                'answer': f"""✅ **Event Finalized!**

Your event **{event_details['title']}** is confirmed for the original time.

📊 **Messages Sent:**
• **Confirmations:** {confirmations_sent} people confirmed for {event_details['time']}
• **Notices:** {notices_sent} people notified about keeping original time

📅 **Final Event Details:**
🎯 **Event:** {event_details['title']}
📅 **Date:** {event_details['date']}
⏰ **Time:** {event_details['time']}
📍 **Location:** {event_details['location']}
👥 **Confirmed Attendees:** {confirmations_sent} people

**Your event is all set! 🎉**"""
            }
            
        except Exception as e:
            print(f"Error in original time confirmation: {str(e)}")
            return {
                'success': False,
                'answer': f"Error confirming original time: {str(e)}"
            }

    def _get_available_attendees(self, event_id):
        """Get list of attendees who are available for the original time"""
        try:
            attendees = []
            
            if self.using_mongodb:
                responses = list(self.db.event_responses.find({
                    'event_id': event_id,
                    'status': 'AVAILABLE'
                }))
                attendees = [{'user_id': resp['user_id']} for resp in responses]
            else:
                conn = sqlite3.connect('social_platform.db')
                c = conn.cursor()
                c.execute('SELECT user_id FROM event_responses WHERE event_id = ? AND status = ?', 
                         (event_id, 'AVAILABLE'))
                attendees = [{'user_id': row[0]} for row in c.fetchall()]
                conn.close()
            
            return attendees
            
        except Exception as e:
            print(f"Error getting available attendees: {str(e)}")
            return []

    def _get_reschedule_requesters(self, event_id):
        """Get list of attendees who requested rescheduling"""
        try:
            requesters = []
            
            if self.using_mongodb:
                responses = list(self.db.event_responses.find({
                    'event_id': event_id,
                    'status': 'RESCHEDULE'
                }))
                requesters = [{'user_id': resp['user_id'], 'requested_time': resp.get('time')} for resp in responses]
            else:
                conn = sqlite3.connect('social_platform.db')
                c = conn.cursor()
                c.execute('SELECT user_id, time FROM event_responses WHERE event_id = ? AND status = ?', 
                         (event_id, 'RESCHEDULE'))
                requesters = [{'user_id': row[0], 'requested_time': row[1]} for row in c.fetchall()]
                conn.close()
            
            return requesters
            
        except Exception as e:
            print(f"Error getting reschedule requesters: {str(e)}")
            return []

    def _set_creator_state(self, creator_id, state, data):
        """Set creator's current state for tracking multi-step processes"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if self.using_mongodb:
                self.db.users.update_one(
                    {'user_id': creator_id},
                    {'$set': {
                        'current_state': state,
                        'state_data': data,
                        'state_updated_at': timestamp
                    }},
                    upsert=True
                )
            else:
                conn = sqlite3.connect('messages.db')  
                c = conn.cursor()
                
                # Create user_states table if it doesn't exist
                c.execute('''
                CREATE TABLE IF NOT EXISTS user_states (
                    user_id INTEGER PRIMARY KEY,
                    current_state TEXT,
                    state_data TEXT,
                    updated_at TEXT
                )
                ''')
                
                c.execute('''
                INSERT OR REPLACE INTO user_states 
                (user_id, current_state, state_data, updated_at)
                VALUES (?, ?, ?, ?)
                ''', (creator_id, state, json.dumps(data), timestamp))
                conn.commit()
                conn.close()
                
        except Exception as e:
            print(f"Error setting creator state: {str(e)}")

    def _clear_creator_state(self, creator_id):
        """Clear creator's current state"""
        try:
            if self.using_mongodb:
                self.db.users.update_one(
                    {'user_id': creator_id},
                    {'$unset': {
                        'current_state': 1,
                        'state_data': 1,
                        'state_updated_at': 1
                    }}
                )
            else:
                conn = sqlite3.connect('messages.db')
                c = conn.cursor()
                c.execute('DELETE FROM user_states WHERE user_id = ?', (creator_id,))
                conn.commit()
                conn.close()
                
        except Exception as e:
            print(f"Error clearing creator state: {str(e)}")


# ===== STANDALONE FUNCTIONS (OUTSIDE THE CLASS) =====

def get_user_state(user_id):
    """Get user's current state from database"""
    try:
        if using_mongodb:
            user_doc = db.users.find_one({'user_id': user_id})
            return user_doc
        else:
            conn = sqlite3.connect('messages.db')
            c = conn.cursor()
            
            # Create user_states table if it doesn't exist
            c.execute('''
            CREATE TABLE IF NOT EXISTS user_states (
                user_id INTEGER PRIMARY KEY,
                current_state TEXT,
                state_data TEXT,
                updated_at TEXT
            )
            ''')
            
            c.execute('SELECT current_state, state_data FROM user_states WHERE user_id = ?', (user_id,))
            result = c.fetchone()
            conn.close()
            
            if result:
                return {
                    'current_state': result[0],
                    'state_data': json.loads(result[1]) if result[1] else {}
                }
            return None
            
    except Exception as e:
        print(f"Error getting user state: {str(e)}")
        return None

        
def generate_enhanced_analytics_with_rescheduling(event_id, creator_id):
    """Generate analytics with intelligent rescheduling recommendations"""
    try:
        # Get basic analytics
        analytics = get_event_response_analytics(event_id)
        
        if not analytics:
            return None
        
        # Initialize the rescheduling assistant
        rescheduling_assistant = IntelligentReschedulingAssistant(client, db, using_mongodb)
        
        # Analyze the rescheduling situation
        rescheduling_analysis = rescheduling_assistant.analyze_rescheduling_situation(analytics)
        
        # Create enhanced summary message
        basic_summary = f"""🎉 **Response Collection Complete!**

Your event received {analytics['total_responded']} out of {analytics['total_invited']} responses ({analytics['response_rate']}% response rate):

✅ **Available**: {analytics['available_count']} people
❌ **Unavailable**: {analytics['unavailable_count']} people  
⏰ **Want to reschedule**: {analytics['reschedule_count']} people
❓ **Unclear/No response**: {analytics['unclear_count']} people"""

        # Add reschedule preferences if any
        if analytics['reschedule_times']:
            basic_summary += "\n\n**Reschedule preferences:**\n"
            for time_pref, count in sorted(analytics['reschedule_times'].items(), key=lambda x: x[1], reverse=True):
                basic_summary += f"• {time_pref}: {count} {'person' if count == 1 else 'people'}\n"
        
        # Add intelligent rescheduling recommendation
        enhanced_summary = basic_summary + f"\n\n🤖 **Smart Recommendation:**\n{rescheduling_analysis['message']}"
        
        # Store the rescheduling analysis in the creator's state
        rescheduling_assistant._set_creator_state(creator_id, 'awaiting_reschedule_decision', {
            'event_id': event_id,
            'analytics': analytics,
            'rescheduling_analysis': rescheduling_analysis,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return {
            'enhanced_analytics': analytics,
            'rescheduling_analysis': rescheduling_analysis,
            'summary_message': enhanced_summary
        }
        
    except Exception as e:
        print(f"Error generating enhanced analytics: {str(e)}")
        return None


def handle_rescheduling_decision_query(user_message, user_id):
    """Handle creator's response to rescheduling decision"""
    try:
        # Check if user is in awaiting_reschedule_decision state
        user_state = get_user_state(user_id)
        
        if not user_state or user_state.get('current_state') != 'awaiting_reschedule_decision':
            return None  # Not in rescheduling decision mode
        
        state_data = user_state.get('state_data', {})
        event_id = state_data.get('event_id')
        analytics = state_data.get('analytics')
        
        if not event_id or not analytics:
            return {
                'success': True,
                'answer': "I couldn't find the event data. Please try again."
            }
        
        # Initialize rescheduling assistant
        rescheduling_assistant = IntelligentReschedulingAssistant(client, db, using_mongodb)
        
        # Handle the creator's decision
        result = rescheduling_assistant.handle_creator_rescheduling_decision(
            user_message, event_id, analytics
        )
        
        return result
        
    except Exception as e:
        print(f"Error handling rescheduling decision: {str(e)}")
        return {
            'success': True,
            'answer': f"Error processing your decision: {str(e)}"
        }


# ===== INITIALIZE THE ASSISTANT =====
rescheduling_assistant = IntelligentReschedulingAssistant(client, db, using_mongodb)

# COMPLETE SmartAttendeeFilteringAssistant - Replace your existing class with this

class SmartAttendeeFilteringAssistant:
    def __init__(self, client, db, using_mongodb):
        self.client = client
        self.db = db
        self.using_mongodb = using_mongodb
        
        # Dedicated system prompt for filtering assistant
        self.system_prompt = """
You are Tarp's Smart Attendee Filtering Assistant. Your job is to help event organizers gather specific information from their confirmed attendees.

CORE CAPABILITIES:
1. Analyze filtering questions from event organizers
2. Determine if questions need attendee responses
3. Rephrase questions for attendees appropriately
4. Process and summarize attendee responses
5. Suggest follow-up filtering questions
6. Detect exit/finalize requests

FILTERING QUESTION TYPES:
- Equipment: "who has a laptop?", "who has a car?", "anyone with a camera?"
- Skills: "who knows Python?", "who can drive?", "who speaks Spanish?"
- Capabilities: "who can lift heavy things?", "who's good at presenting?"
- Preferences: "who likes spicy food?", "who prefers morning events?"
- Availability: "who can stay late?", "who's free tomorrow?"
- Experience: "who has done this before?", "who's a beginner?"

EXIT/FINALIZE DETECTION:
If the organizer wants to stop filtering, look for phrases like:
- "that's enough", "done", "finish", "finalize", "create event", "proceed"
- "no more questions", "I'm good", "ready to go", "let's proceed"

RESPONSE FORMAT:
Always respond with JSON:
{
    "action": "send_to_attendees|provide_info|clarify_question|exit_filtering",
    "analysis": {
        "question_type": "equipment|skill|preference|availability|capability|experience|exit|other",
        "needs_attendee_responses": true/false,
        "attendee_question": "rephrased question for attendees",
        "expected_response_type": "yes_no|details|multiple_choice|numeric",
        "reasoning": "why this approach"
    },
    "message": "response message to the organizer",
    "attendee_message": "message to send to attendees (if applicable)"
}
"""

    def process_filtering_query(self, user_message, filtering_state):
        """Process a filtering query from an event organizer"""
        try:
            # Build context for the LLM
            context = f"""
FILTERING CONTEXT:
- Event ID: {filtering_state.get('event_id')}
- Total confirmed attendees: {filtering_state.get('total_available', 0)}
- Organizer question: "{user_message}"

Analyze this filtering question and determine the best approach.
"""

            # Call the dedicated filtering LLM
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            print(f"Filtering LLM Response: {response_text}")
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "action": "send_to_attendees",
                    "analysis": {
                        "question_type": "other",
                        "needs_attendee_responses": True,
                        "attendee_question": user_message,
                        "expected_response_type": "details",
                        "reasoning": "JSON parsing failed, defaulting to send to attendees"
                    },
                    "message": f"I'll send your question '{user_message}' to confirmed attendees.",
                    "attendee_message": f"The event organizer asks: {user_message}. Please respond with your answer."
                }
            
            return result
            
        except Exception as e:
            print(f"Error in Filtering LLM: {str(e)}")
            return {
                "action": "send_to_attendees",
                "analysis": {
                    "question_type": "other",
                    "needs_attendee_responses": True,
                    "attendee_question": user_message,
                    "expected_response_type": "details",
                    "reasoning": f"Error occurred: {str(e)}"
                },
                "message": f"I'll send your question to attendees.",
                "attendee_message": f"Question from organizer: {user_message}"
            }

    def handle_filtering_request(self, user_message, user_id, filtering_state):
        """Main handler for filtering requests"""
        try:
            # Use the filtering LLM to analyze the request
            llm_result = self.process_filtering_query(user_message, filtering_state)
            
            action = llm_result.get('action', 'send_to_attendees')
            
            if action == 'exit_filtering':
                return self._exit_filtering_mode(user_message, user_id, filtering_state, llm_result)
            elif action == 'send_to_attendees':
                return self._send_question_to_attendees(user_message, user_id, filtering_state, llm_result)
            elif action == 'provide_info':
                return self._provide_direct_info(user_message, user_id, filtering_state, llm_result)
            elif action == 'clarify_question':
                return self._request_clarification(user_message, user_id, filtering_state, llm_result)
            else:
                return self._send_question_to_attendees(user_message, user_id, filtering_state, llm_result)
                
        except Exception as e:
            print(f"Error in filtering handler: {str(e)}")
            return {
                'success': True,
                'answer': "I'm having trouble processing your filtering question. Please try again."
            }

    def _send_question_to_attendees(self, user_message, user_id, filtering_state, llm_result):
        """Send filtering question to attendees based on LLM analysis"""
        try:
            event_id = filtering_state['event_id']
            available_attendees = filtering_state['available_attendees']
            
            # Get event details
            event_details = get_event_details(event_id)
            if not event_details:
                return {'success': True, 'answer': "Could not retrieve event details for filtering."}
            
            # Use LLM-generated attendee message
            attendee_message = llm_result.get('attendee_message', f"Question: {user_message}")
            
            # Create complete message for attendees
            full_message = f"""🔍 **Question from Event Organizer**

The organizer of **{event_details['title']}** (scheduled for {event_details['date']} at {event_details['time']}) has a question for confirmed attendees:

{attendee_message}

Your response will help the organizer better plan the event."""

            # Create filter ID for tracking
            import uuid
            filter_id = f"filter_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Send messages to attendees
            messages_sent = 0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for attendee_id in available_attendees:
                try:
                    if self.using_mongodb:
                        filter_message_doc = {
                            'from_user_id': user_id,
                            'to_user_id': attendee_id,
                            'message_type': 'attendee_filter_question',
                            'content': full_message,
                            'event_id': event_id,
                            'filter_id': filter_id,
                            'original_question': user_message,
                            'llm_analysis': llm_result['analysis'],
                            'expected_response_type': llm_result['analysis'].get('expected_response_type', 'details'),
                            'event_details': event_details,
                            'timestamp': timestamp,
                            'is_read': 0,
                            'response': None
                        }
                        
                        self.db.messages.insert_one(filter_message_doc)
                        messages_sent += 1
                        
                except Exception as e:
                    print(f"Error sending filtering message to user {attendee_id}: {e}")
            
            # Update filtering state
            filtering_key = f"attendee_filtering_{user_id}"
            from flask import session
            
            if filtering_key in session:
                session[filtering_key]['current_filter'] = {
                    'filter_id': filter_id,
                    'question': user_message,
                    'llm_analysis': llm_result['analysis'],
                    'sent_to': available_attendees,
                    'messages_sent': messages_sent,
                    'created_at': timestamp,
                    'waiting_for_responses': True
                }
                session.modified = True
            
            return {
                'success': True,
                'answer': f"""🤖 **Smart Filtering Question Sent!**

{llm_result.get('message', 'Question sent to attendees')}

📧 **Messages Sent:** {messages_sent} confirmed attendees
🎯 **Question Type:** {llm_result['analysis'].get('question_type', 'general').title()}
📋 **Expected Responses:** {llm_result['analysis'].get('expected_response_type', 'details').replace('_', ' ').title()}

⏳ **What happens next:**
• Attendees will receive your question
• They'll respond with their answers  
• I'll collect and analyze all responses
• You'll get an intelligent summary with decision options

🧠 **AI Analysis:** {llm_result['analysis'].get('reasoning', 'Question processed intelligently')}

🔄 **Status:** Waiting for responses..."""
            }
            
        except Exception as e:
            print(f"Error sending attendee messages: {str(e)}")
            return {'success': True, 'answer': f"Error sending filtering question: {str(e)}"}

    def collect_filtering_responses(self, filter_id, creator_id):
        """Collect all responses for a filter and send results to creator"""
        try:
            # Get all responses for this filter
            responses = self._get_filtering_responses(filter_id)
            original_question = self._get_original_filtering_question(filter_id)
            
            if not responses or not original_question:
                return False
            
            # Analyze responses using LLM
            analysis_result = self.analyze_filtering_responses(responses, original_question)
            
            # Send results and decision options to creator
            self.send_filtering_results_to_creator(creator_id, analysis_result, filter_id)
            
            return True
            
        except Exception as e:
            print(f"Error in filtering response collection: {str(e)}")
            return False

    def analyze_filtering_responses(self, responses, original_question):
        """Use LLM to analyze filtering responses and categorize attendees"""
        try:
            # Build context for LLM analysis
            responses_text = ""
            for response in responses:
                responses_text += f"• User #{response['user_id']}: \"{response['response_text']}\"\n"
            
            analysis_prompt = f"""
Analyze these responses to the filtering question and categorize attendees:

**Original Question:** "{original_question}"

**Responses:**
{responses_text}

Based on the responses, categorize attendees into:
1. MATCH - Attendees who satisfy the requirement/question
2. NO_MATCH - Attendees who don't satisfy the requirement
3. UNCLEAR - Unclear or ambiguous responses

Respond with JSON:
{{
    "question": "{original_question}",
    "total_responses": {len(responses)},
    "categories": {{
        "match": [
            {{"user_id": 123, "response": "their response", "reason": "why they match"}}
        ],
        "no_match": [
            {{"user_id": 456, "response": "their response", "reason": "why they don't match"}}
        ],
        "unclear": [
            {{"user_id": 789, "response": "their response", "reason": "why unclear"}}
        ]
    }},
    "summary": {{
        "match_count": 3,
        "no_match_count": 1,
        "unclear_count": 0
    }},
    "recommendation": "proceed_with_matches|keep_everyone|ask_clarification",
    "reasoning": "explanation of recommendation"
}}
"""

            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing survey responses and making smart recommendations."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            print(f"Filtering Analysis LLM Response: {response_text}")
            
            try:
                analysis = json.loads(response_text)
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return self._create_fallback_analysis(responses, original_question)
                
        except Exception as e:
            print(f"Error in LLM filtering analysis: {str(e)}")
            return self._create_fallback_analysis(responses, original_question)

    def send_filtering_results_to_creator(self, creator_id, analysis_result, filter_id):
        """Send filtering results and decision options to the event creator"""
        try:
            # Build the results message
            question = analysis_result['question']
            summary = analysis_result['summary']
            categories = analysis_result['categories']
            recommendation = analysis_result.get('recommendation', 'proceed_with_matches')
            reasoning = analysis_result.get('reasoning', 'Based on the responses')
            
            # Create detailed results message
            results_message = f"""📊 **Filtering Results: "{question}"**

"""
            
            # Add matches
            if categories['match']:
                results_message += f"✅ **Match the requirement:** {summary['match_count']} people\n"
                for match in categories['match']:
                    results_message += f"• User #{match['user_id']}: \"{match['response']}\"\n"
                results_message += "\n"
            
            # Add no matches  
            if categories['no_match']:
                results_message += f"❌ **Don't match:** {summary['no_match_count']} people\n"
                for no_match in categories['no_match']:
                    results_message += f"• User #{no_match['user_id']}: \"{no_match['response']}\"\n"
                results_message += "\n"
            
            # Add unclear
            if categories['unclear']:
                results_message += f"❓ **Unclear responses:** {summary['unclear_count']} people\n"
                for unclear in categories['unclear']:
                    results_message += f"• User #{unclear['user_id']}: \"{unclear['response']}\"\n"
                results_message += "\n"
            
            # Add intelligent decision options
            results_message += f"""🤖 **Smart Recommendation:** {reasoning}

**What would you like to do?**

1️⃣ **Proceed with matches only** - Continue with {summary['match_count']} people who meet the requirement
2️⃣ **Keep everyone** - Continue with all {summary['match_count'] + summary['no_match_count']} confirmed attendees  
3️⃣ **Ask another question** - Filter further with additional criteria
4️⃣ **Done filtering** - Finalize event with current group

**Example responses:**
• "proceed with matches" or "1"
• "keep everyone" or "2"  
• "ask about experience level" or "3"
• "done" or "4"
"""

            # Send message to creator
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if self.using_mongodb:
                results_message_doc = {
                    'from_user_id': 0,  # System message
                    'to_user_id': creator_id,
                    'message_type': 'filtering_results',
                    'content': results_message,
                    'filter_id': filter_id,
                    'analysis_data': analysis_result,
                    'timestamp': timestamp,
                    'is_read': 0,
                    'awaiting_decision': True
                }
                
                self.db.messages.insert_one(results_message_doc)
            
            print(f"📊 FILTERING RESULTS: Sent to creator {creator_id} for filter {filter_id}")
            
        except Exception as e:
            print(f"Error sending filtering results: {str(e)}")

    def _provide_direct_info(self, user_message, user_id, filtering_state, llm_result):
        """Provide direct information without needing attendee responses"""
        available_count = filtering_state.get('total_available', 0)
        
        return {
            'success': True,
            'answer': f"""🤖 **Smart Analysis**

{llm_result.get('message', "Here's the information you requested")}

📊 **Current Status:**
• **Confirmed Attendees:** {available_count} people
• **Question Type:** {llm_result['analysis'].get('question_type', 'informational').title()}

🧠 **AI Analysis:** {llm_result['analysis'].get('reasoning', 'This question can be answered with available data')}

💡 **Need more specific info?** Try asking questions like:
• "Do you have [specific item]?"
• "Can you [specific skill/ability]?"
• "Are you available for [specific time]?"
"""
        }

    def _request_clarification(self, user_message, user_id, filtering_state, llm_result):
        """Request clarification from the organizer"""
        return {
            'success': True,
            'answer': f"""🤖 **Need Clarification**

{llm_result.get('message', 'I need more details to help you effectively')}

🧠 **AI Analysis:** {llm_result['analysis'].get('reasoning', 'The question needs clarification')}

💡 **Suggestions:**
• Be more specific about what you're looking for
• Ask direct yes/no questions for clearer responses
• Specify exactly what information you need

**Example:** Instead of "who's good with tech?" try "who knows Python programming?" or "who has experience with video editing?"
"""
        }

    def _exit_filtering_mode(self, user_message, user_id, filtering_state, llm_result):
        """Handle exit from filtering mode and finalize event"""
        try:
            event_id = filtering_state['event_id']
            available_attendees = filtering_state['available_attendees']
            total_available = filtering_state.get('total_available', len(available_attendees))
            
            # Get event details
            event_details = get_event_details(event_id)
            if not event_details:
                return {'success': True, 'answer': "Could not retrieve event details."}
            
            # Clear filtering state from session
            filtering_key = f"attendee_filtering_{user_id}"
            from flask import session
            
            if filtering_key in session:
                session.pop(filtering_key, None)
                session.modified = True
                print(f"🎯 FILTERING: User {user_id} exited filtering mode")
            
            return {
                'success': True,
                'answer': f"""✅ **Event Finalized!**

{llm_result.get('message', 'Perfect! Your event is now finalized.')}

📅 **Final Event Details:**
🎯 **Event:** {event_details['title']}
📅 **Date:** {event_details['date']}
⏰ **Time:** {event_details['time']}
📍 **Location:** {event_details['location']}
👥 **Confirmed Attendees:** {total_available} people

🎉 **Your event is ready to go!**"""
            }
            
        except Exception as e:
            print(f"Error exiting filtering mode: {str(e)}")
            return {
                'success': True,
                'answer': f"Event finalized successfully!"
            }

    def _get_filtering_responses(self, filter_id):
        """Get all responses for a specific filter"""
        try:
            if self.using_mongodb:
                responses = list(self.db.filtering_responses.find({'filter_id': filter_id}))
            else:
                conn = sqlite3.connect('social_platform.db')
                c = conn.cursor()
                c.execute('SELECT * FROM filtering_responses WHERE filter_id = ?', (filter_id,))
                responses = [dict(row) for row in c.fetchall()]
                conn.close()
            return responses
        except Exception as e:
            print(f"Error getting filtering responses: {str(e)}")
            return []

    def _get_original_filtering_question(self, filter_id):
        """Get the original question for a filter"""
        try:
            if self.using_mongodb:
                filter_msg = self.db.messages.find_one({'filter_id': filter_id, 'message_type': 'attendee_filter_question'})
                return filter_msg.get('original_question') if filter_msg else None
            else:
                conn = sqlite3.connect('messages.db')
                c = conn.cursor()
                c.execute('SELECT original_question FROM messages WHERE filter_id = ? AND message_type = ?', 
                         (filter_id, 'attendee_filter_question'))
                result = c.fetchone()
                conn.close()
                return result[0] if result else None
        except Exception as e:
            print(f"Error getting original question: {str(e)}")
            return None

    def _create_fallback_analysis(self, responses, original_question):
        """Create fallback analysis if LLM fails"""
        return {
            "question": original_question,
            "total_responses": len(responses),
            "categories": {
                "match": [],
                "no_match": [],
                "unclear": [{"user_id": resp['user_id'], "response": resp['response_text'], "reason": "fallback analysis"} for resp in responses]
            },
            "summary": {
                "match_count": 0,
                "no_match_count": 0,
                "unclear_count": len(responses)
            },
            "recommendation": "keep_everyone",
            "reasoning": "Could not analyze responses automatically"
        }

# Initialize the smart filtering assistant
smart_filtering_assistant = SmartAttendeeFilteringAssistant(client, db, using_mongodb)

# Add this new class to your paste.txt file




# Update your find_matches route to use smart recommendations
def enhance_find_matches_with_smart_recommendations():
    """
    This function shows how to modify your existing find_matches route
    to include smart recommendations when no results are found
    """
    
    # In your existing @app.route('/find_matches', methods=['POST']) function,
    # replace the "No results found" section with this:
    
    # After you've processed the query and determined there are no results:
    if not has_results:  # This is your existing check
        user_message = request.json.get('original_query', '')
        
        # Extract the searched interest from the query
        searched_interest = None
        if query_type == 'interest_search':
            interests = entities.get('interests', [])
            if interests:
                searched_interest = interests[0]  # Take the first interest
        
        if searched_interest:
            # Get smart recommendations using the new assistant
            recommendations = smart_recommender.get_smart_recommendations(searched_interest, results)
            
            if recommendations:
                # Use the LLM-generated recommendation message
                response_prompt = f"""
User searched for: {user_message}

{recommendations['recommendation_message']}

Available interest categories: {', '.join(INTEREST_CATEGORIES.keys())}

Use the recommendation message provided above, but make it conversational and helpful.
Don't suggest interests that weren't mentioned in the recommendation message.
"""
            else:
                # Fallback to category suggestions
                response_prompt = f"""
User searched for: {user_message}

No results found in the database for this specific interest.

Available interest categories: {', '.join(INTEREST_CATEGORIES.keys())}

Be honest that the specific interest searched for is not in the database.
Suggest browsing the available categories instead.
"""
        else:
            # Original fallback logic for non-interest searches
            response_prompt = f"""
User searched for: {user_message}

No results found in the database.

Available categories: {', '.join(INTEREST_CATEGORIES.keys())}

Respond helpfully but honestly about what's available.
"""


# Example usage in your response generation:
def generate_enhanced_response_with_smart_recommendations(user_message, results, query_type, entities):
    """
    Enhanced response generation that includes smart recommendations
    """
    
    # Check if we have any results
    has_results = False
    if results:
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 0:
                        has_results = True
                        break
            elif isinstance(value, list) and len(value) > 0:
                has_results = True
                break
            if has_results:
                break
    
    if not has_results and query_type == 'interest_search':
        # Extract the searched interest
        interests = entities.get('interests', [])
        if interests:
            searched_interest = interests[0]
            
            # Get smart recommendations
            recommendations = smart_recommender.get_smart_recommendations(searched_interest, results)
            
            if recommendations and recommendations['similar_matches']:
                # We have smart recommendations
                verified_interests = smart_recommender.verify_interests_exist(recommendations['similar_matches'])
                
                if verified_interests:
                    response_message = f"""I couldn't find users interested in "{searched_interest}" in our current database.

However, I found some related interests that we do have:
{', '.join(f'• {interest}' for interest in verified_interests)}

{recommendations['explanation']}

Would you like me to search for people interested in any of these related topics instead?"""
                else:
                    response_message = f"""I couldn't find users interested in "{searched_interest}" in our current database, and unfortunately we don't have any closely related interests available either.

You could try browsing our available categories: {', '.join(INTEREST_CATEGORIES.keys())}"""
            else:
                response_message = f"""I couldn't find users interested in "{searched_interest}" in our current database, and we don't have any closely related interests available.

You could try browsing our available categories: {', '.join(INTEREST_CATEGORIES.keys())}"""
            
            return response_message
    
    # For other cases, use your existing logic
    return "Standard response generation logic here..."

# Updated handle_attendee_filtering function
def handle_attendee_filtering(user_message, user_id, filtering_state):
    """Enhanced attendee filtering using dedicated LLM assistant"""
    return smart_filtering_assistant.handle_filtering_request(user_message, user_id, filtering_state)
class InterestCaptureManager:
    """
    A class that manages capturing, storing, and retrieving user interests
    from conversations
    """
    def __init__(self, client, db=None, using_mongodb=False):
        self.client = client
        self.db = db
        self.using_mongodb = using_mongodb
        
        # System prompt for interest extraction and analysis
        self.system_prompt = """
You are an AI designed to understand the context of conversations and identify user interests, topics, or intentions.

Your task:
1. Identify the primary topic/interest being discussed in this conversation
2. Create a concise name/title for this interest or topic
3. Summarize what the user seems to want or need
4. Extract any key details relevant to this interest/topic
5. Determine if this represents an ongoing interest vs a one-time query

Do NOT limit your analysis to predefined categories. Be open to identifying ANY topic of interest.

Respond ONLY with a JSON object containing:
{
    "interest_title": "A short descriptive name for this topic/interest",
    "context_summary": "Brief 1-2 sentence summary of what's being discussed",
    "user_intent": "What the user appears to want or need",
    "key_details": ["list", "of", "relevant", "details"],
    "is_ongoing_interest": true/false,
    "confidence": "high/medium/low"
}
"""

    def analyze_conversation(self, conversation_history, user_message):
        """
        Analyze conversation to understand context, extract interests, and determine user intent
        Returns a structured analysis of the conversation
        """
        try:
            # Prepare the conversation context for the LLM
            conversation_context = ""
            
            # Include up to 5 recent exchanges for context
            recent_exchanges = []
            for i in range(min(10, len(conversation_history)), 0, -2):
                if i-1 >= 0:
                    user_msg = conversation_history[i-1]['content']
                    if i < len(conversation_history):
                        assistant_msg = conversation_history[i]['content']
                        recent_exchanges.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
            
            # Add the recent message
            recent_exchanges.append(f"User: {user_message}")
            
            # Combine exchanges
            conversation_context = "\n\n".join(recent_exchanges)
            
            analysis_prompt = f"""
CONVERSATION HISTORY:
{conversation_context}

Analyze this conversation to identify the user's interests, intentions, and needs.
"""

            # Call the LLM for context analysis
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON if wrapped in text
                json_match = re.search(r'(\{.*\})', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    result = {
                        "interest_title": "General Conversation",
                        "context_summary": "Could not analyze the conversation context",
                        "user_intent": "Unknown",
                        "key_details": [],
                        "is_ongoing_interest": False,
                        "confidence": "low"
                    }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing conversation context: {str(e)}")
            return {
                "interest_title": "General Conversation",
                "context_summary": f"Error analyzing conversation: {str(e)}",
                "user_intent": "Unknown",
                "key_details": [],
                "is_ongoing_interest": False,
                "confidence": "low"
            }

    def store_interest(self, user_id, context_analysis):
       
        try:
            if not user_id:
                print("No user_id provided, can't store interest")
                return None
                
            #interest record
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            interest_id = f"interest_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            interest_record = {
                'interest_id': interest_id,
                'user_id': user_id,
                'interest_title': context_analysis.get('interest_title'),
                'context_summary': context_analysis.get('context_summary'),
                'user_intent': context_analysis.get('user_intent'),
                'key_details': context_analysis.get('key_details', []),
                'is_ongoing_interest': context_analysis.get('is_ongoing_interest', False),
                'confidence': context_analysis.get('confidence', 'low'),
                'created_at': timestamp
            }
            
            # Add debug info
            print(f"DB Info - using_mongodb: {self.using_mongodb}, db is None: {self.db is None}")
            
            # Store in the database
            if self.using_mongodb and self.db is not None:
                # MongoDB version
                print(f"Attempting to store in MongoDB collection: user_interests")
                result = self.db.user_interests.insert_one(interest_record)
                print(f"MongoDB insert result: {result.inserted_id}")
            else:
                # SQLite version
                print(f"Storing in SQLite database: social_platform.db")
                self._ensure_sqlite_table()
                conn = sqlite3.connect('social_platform.db')
                c = conn.cursor()
                
                # Insert the interest
                c.execute('''
                INSERT INTO user_interests
                (interest_id, user_id, interest_title, context_summary, user_intent, 
                key_details, is_ongoing_interest, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    interest_record['interest_id'],
                    interest_record['user_id'],
                    interest_record['interest_title'],
                    interest_record['context_summary'],
                    interest_record['user_intent'],
                    json.dumps(interest_record['key_details']),
                    1 if interest_record['is_ongoing_interest'] else 0,
                    interest_record['confidence'],
                    interest_record['created_at']
                ))
                
                conn.commit()
                conn.close()
                print("SQLite insert completed")
                
            print(f"Successfully stored interest for user {user_id}: {interest_record['interest_title']} with ID: {interest_id}")
            return interest_id
            
        except Exception as e:
            print(f"Error storing user interest: {str(e)}")
            import traceback
            traceback.print_exc()  # Print stack trace 
            return None

   
    def get_interest_by_id(self, interest_id):
        """
        Get a specific interest by ID
        """
        try:
            if not interest_id:
                return None
                
            # Retrieve from database
            if self.using_mongodb and self.db:
                # MongoDB version
                interest = self.db.user_interests.find_one({'interest_id': interest_id})
                
                # Convert ObjectId to string
                if interest and '_id' in interest:
                    interest['_id'] = str(interest['_id'])
                    
                return interest
            else:
                # SQLite version
                self._ensure_sqlite_table()
                conn = sqlite3.connect('social_platform.db')
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                c.execute('''
                SELECT * FROM user_interests
                WHERE interest_id = ?
                ''', (interest_id,))
                
                result = c.fetchone()
                conn.close()
                
                if not result:
                    return None
                    
                # Convert to dict
                interest = dict(result)
                # Parse JSON string back to list
                interest['key_details'] = json.loads(interest['key_details'])
                # Convert boolean
                interest['is_ongoing_interest'] = bool(interest['is_ongoing_interest'])
                
                return interest
                
        except Exception as e:
            print(f"Error getting interest by ID: {str(e)}")
            return None

    def delete_interest(self, interest_id, user_id=None):
        """
        Delete a specific interest
        """
        try:
            if not interest_id:
                return False
                
            # Delete from database
            if self.using_mongodb and self.db:
                # MongoDB version
                query = {'interest_id': interest_id}
                if user_id:
                    query['user_id'] = user_id
                    
                result = self.db.user_interests.delete_one(query)
                return result.deleted_count > 0
            else:
                # SQLite version
                self._ensure_sqlite_table()
                conn = sqlite3.connect('social_platform.db')
                c = conn.cursor()
                
                if user_id:
                    c.execute('''
                    DELETE FROM user_interests
                    WHERE interest_id = ? AND user_id = ?
                    ''', (interest_id, user_id))
                else:
                    c.execute('''
                    DELETE FROM user_interests
                    WHERE interest_id = ?
                    ''', (interest_id,))
                
                deleted = c.rowcount > 0
                conn.commit()
                conn.close()
                
                return deleted
                
        except Exception as e:
            print(f"Error deleting interest: {str(e)}")
            return False

    def search_interests(self, user_id, query):
        """
        Search for interests matching a keyword
        """
        try:
            if not user_id or not query:
                return []
                
            search_term = query.lower()
                
            # Search from database
            if self.using_mongodb and self.db:
                # MongoDB version - text search
                interests = list(self.db.user_interests.find({
                    'user_id': user_id,
                    '$or': [
                        {'interest_title': {'$regex': search_term, '$options': 'i'}},
                        {'context_summary': {'$regex': search_term, '$options': 'i'}},
                        {'user_intent': {'$regex': search_term, '$options': 'i'}}
                    ]
                }))
                
                # Convert ObjectId to string
                for interest in interests:
                    if '_id' in interest:
                        interest['_id'] = str(interest['_id'])
                        
                return interests
            else:
                # SQLite version
                self._ensure_sqlite_table()
                conn = sqlite3.connect('social_platform.db')
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                c.execute('''
                SELECT * FROM user_interests
                WHERE user_id = ? AND (
                    interest_title LIKE ? OR
                    context_summary LIKE ? OR
                    user_intent LIKE ?
                )
                ORDER BY created_at DESC
                ''', (user_id, f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
                
                results = c.fetchall()
                conn.close()
                
                # Convert to list of dicts
                interests = []
                for row in results:
                    interest = dict(row)
                    # Parse JSON string back to list
                    interest['key_details'] = json.loads(interest['key_details'])
                    # Convert boolean
                    interest['is_ongoing_interest'] = bool(interest['is_ongoing_interest'])
                    interests.append(interest)
                    
                return interests
                
        except Exception as e:
            print(f"Error searching interests: {str(e)}")
            return []

    def _ensure_sqlite_table(self):
        """
        Create the SQLite table if it doesn't exist
        """
        try:
            conn = sqlite3.connect('social_platform.db')
            c = conn.cursor()
            
            # Create table if it doesn't exist
            c.execute('''
            CREATE TABLE IF NOT EXISTS user_interests (
                interest_id TEXT PRIMARY KEY,
                user_id INTEGER,
                interest_title TEXT,
                context_summary TEXT,
                user_intent TEXT,
                key_details TEXT,
                is_ongoing_interest INTEGER,
                confidence TEXT,
                created_at TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error ensuring SQLite table: {str(e)}")
interest_capture = InterestCaptureManager(client, db, using_mongodb)
    
def process_chat_query(user_message, filename=None, conversation_history=None, user_id=None, request=None):
    """Handle LLM queries using the three-layer approach with conversation history + Enhanced Filtering System"""
    try:
        # session key that's unique to this user/conversation
        session_key = f"event_creation_state_{user_id}" if user_id else "event_creation_state"
        
        
        print(f"🔍 CONFIRMATION DEBUG: user_message='{user_message}'")
        print(f"🔍 CONFIRMATION DEBUG: user_id={user_id}")
        print(f"🔍 CONFIRMATION DEBUG: session_key='{session_key}'")
        print(f"🔍 CONFIRMATION DEBUG: all session keys={list(session.keys())}")
        
        # Get session state
        event_creation_state = session.get(session_key, {})
        in_event_creation = bool(event_creation_state)
        current_step = event_creation_state.get('step', '') if event_creation_state else ''
        
        print(f"🔍 CONFIRMATION DEBUG: event_creation_state={event_creation_state}")
        print(f"🔍 CONFIRMATION DEBUG: in_event_creation={in_event_creation}")
        print(f"🔍 CONFIRMATION DEBUG: current_step='{current_step}'")
        
        # ===== ABSOLUTE PRIORITY: CONFIRMATION STEP DETECTION =====
        # This MUST be the first check - no exceptions
        if in_event_creation and current_step in ['confirm', 'confirm_details']:
            print(f"🎯 CONFIRMATION DETECTED: User is in confirmation step with message '{user_message}'")
            
            affirmative_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'confirm', 'go ahead', 'do it', 'create it', 'yup', 'y']
            negative_words = ['no', 'nope', 'cancel', 'stop', 'abort', 'don\'t', 'never mind', 'n']
            
            user_message_lower = user_message.lower().strip()
            
            if any(word == user_message_lower or word in user_message_lower for word in affirmative_words):
                print("✅ CONFIRMATION: User said YES - creating event")
                return smart_event_assistant.handle_smart_event_creation(
                    'event_creation', user_message, {'confirmation': 'yes'}, event_creation_state, session_key, user_id
                )
            elif any(word == user_message_lower or word in user_message_lower for word in negative_words):
                print("❌ CONFIRMATION: User said NO - cancelling event")
                return smart_event_assistant.handle_smart_event_creation(
                    'event_creation', user_message, {'confirmation': 'no'}, event_creation_state, session_key, user_id
                )
            else:
                print("❓ CONFIRMATION: Unclear response - asking for clarification")
                return smart_event_assistant.handle_smart_event_creation(
                    'event_creation', user_message, {'confirmation': 'unclear'}, event_creation_state, session_key, user_id
                )
        
        # ===== ENHANCED CONTEXT DETECTION =====
        filtering_key = f"attendee_filtering_{user_id}"
        waiting_key = f"waiting_for_responses_{user_id}"
        
        # 1. Checks rescheduling decision (time-sensitive)
        rescheduling_result = handle_rescheduling_decision_query(user_message, user_id)
        if rescheduling_result is not None:
            print(f"User {user_id} is making a rescheduling decision")
            return rescheduling_result
        
        # 2. Checks if user is in attendee filtering mode
        if filtering_key in session:
            print(f"User {user_id} is in attendee filtering mode")
            return handle_attendee_filtering(user_message, user_id, session[filtering_key])
        
        # 3. Checks if user is waiting for event responses 
        if waiting_key in session:
            waiting_state = session[waiting_key]
            deadline = datetime.strptime(waiting_state['deadline'], "%Y-%m-%d %H:%M:%S")
            
            if datetime.now() > deadline:
                print(f"15-minute deadline passed for event {waiting_state['event_id']}")
                return handle_response_deadline_reached(user_id, waiting_state)
            else:
                # Still waiting - show detailed status
                remaining_time = deadline - datetime.now()
                minutes_left = int(remaining_time.total_seconds() / 60)
                seconds_left = int(remaining_time.total_seconds() % 60)
                
                event_details = waiting_state.get('event_details', {})
                invitations_sent = waiting_state.get('invitations_sent', 0)
                
                # Get current response count
                current_responses = 0
                responses_breakdown = {'available': 0, 'unavailable': 0, 'maybe': 0}
                
                try:
                    if using_mongodb:
                        current_responses = db.event_responses.count_documents({
                            'event_id': waiting_state['event_id']
                        })
                        
                        pipeline = [
                            {'$match': {'event_id': waiting_state['event_id']}},
                            {'$group': {
                                '_id': '$status',  
                                'count': {'$sum': 1}
                            }}
                        ]
                        breakdown_results = list(db.event_responses.aggregate(pipeline))
                        for result in breakdown_results:
                            response_type = result['_id'].lower()
                            if response_type == 'available':
                                responses_breakdown['available'] = result['count']
                            elif response_type == 'unavailable':
                                responses_breakdown['unavailable'] = result['count']
                            elif response_type == 'reschedule':
                                responses_breakdown['maybe'] = result['count']
                                
                    else:
                        # SQLite query for response count
                        conn = sqlite3.connect('social_platform.db')
                        c = conn.cursor()
                        
                        c.execute('SELECT COUNT(*) FROM event_responses WHERE event_id = ?', 
                                (waiting_state['event_id'],))
                        result = c.fetchone()
                        current_responses = result[0] if result else 0
                        
                        c.execute('''
                            SELECT status, COUNT(*) 
                            FROM event_responses 
                            WHERE event_id = ? 
                            GROUP BY status
                        ''', (waiting_state['event_id'],))
                        
                        breakdown_results = c.fetchall()
                        for status, count in breakdown_results:
                            if status.lower() == 'available':
                                responses_breakdown['available'] = count
                            elif status.lower() == 'unavailable':
                                responses_breakdown['unavailable'] = count
                            elif status.lower() == 'reschedule':
                                responses_breakdown['maybe'] = count
                        
                        conn.close()
                        
                except Exception as e:
                    print(f"Error getting current responses: {e}")
                
                response_rate = (current_responses / invitations_sent * 100) if invitations_sent > 0 else 0
                
                waiting_message = f"""⏳ **Response Collection in Progress**

📅 **Event:** {event_details.get('title', 'Your Event')}
📍 **Location:** {event_details.get('location', 'TBD')}
🗓️ **Date & Time:** {event_details.get('date', 'TBD')} at {event_details.get('time', 'TBD')}

📧 **Invitations Sent:** {invitations_sent} users
📬 **Responses So Far:** {current_responses} received ({response_rate:.1f}% response rate)

📊 **Response Breakdown:**
✅ Available: {responses_breakdown['available']}
❌ Unavailable: {responses_breakdown['unavailable']}
🤔 Maybe: {responses_breakdown['maybe']}

⏰ **Time Remaining:** {minutes_left} minutes, {seconds_left} seconds

🔄 **What's happening:**
• Users are receiving and responding to your event invitation
• I'm collecting all responses in real-time
• You'll get complete analytics automatically when the 15-minute period ends

✨ **Just sit back and relax - I'll handle everything!**"""

                return {
                    'success': True,
                    'answer': waiting_message,
                    'waiting_status': {
                        'event_id': waiting_state['event_id'],
                        'time_remaining_seconds': int(remaining_time.total_seconds()),
                        'responses_received': current_responses,
                        'invitations_sent': invitations_sent,
                        'response_rate': response_rate,
                        'breakdown': responses_breakdown
                    }
                }
        
        # LAYER 1: Query Classification
        print(f"Processing query: {user_message}")
        
        # Enhanced override logic for event creation steps (excluding confirmation)
        should_override_classification = (
            in_event_creation and 
            current_step in ['date', 'time', 'location'] and  # NOT including 'confirm'
            len(user_message.split()) <= 5
        )
        
        if should_override_classification:
            print("In event creation flow, maintaining classification as event_creation")
            classification = {"query_type": "event_creation", "entities": {}}
        else:
            # Normal classification  call the LLM
            print("Calling LLM for query classification...")
            classifier_response = client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2
            )
            
            classification_text = classifier_response.choices[0].message.content
            print(f"LLM classification response: {classification_text}")
            
            try:
                classification = json.loads(classification_text)
            except json.JSONDecodeError:
                json_match = re.search(r'(\{.*\})', classification_text, re.DOTALL)
                if json_match:
                    try:
                        classification = json.loads(json_match.group(1))
                    except:
                        if in_event_creation:
                            classification = {"query_type": "event_creation", "entities": {}}
                        else:
                            classification = {"query_type": "general", "entities": {}}
                else:
                    if in_event_creation:
                        classification = {"query_type": "event_creation", "entities": {}}
                    else:
                        classification = {"query_type": "general", "entities": {}}
        
        print(f"Query classified as: {classification}")
        
        # Get the query type and entities
        query_type = classification.get('query_type', '')
        entities = classification.get('entities', {})
        
        print(f"Processing query of type '{query_type}' with entities: {entities}")
        
        # HANDLE DIFFERENT QUERY TYPES
        
        if query_type == 'general_knowledge':
            print("General knowledge question detected, analyzing for interests")
            
            # Extracts the interests and create summary
            context_analysis = interest_capture.analyze_conversation(conversation_history, user_message)
            print(f"Interest analysis: {context_analysis}")
            
            # Stores the interest if confidence is medium or high
            interest_id = None
            if context_analysis.get('confidence', 'low') in ['medium', 'high'] and user_id:
                interest_id = interest_capture.store_interest(user_id, context_analysis)
                print(f"Stored interest with ID: {interest_id}")
            
            # Process general knowledge response (your existing code)
            general_knowledge_prompt = """
            You are an AI assistant providing helpful, accurate, and thoughtful responses to general knowledge questions.
            
            When responding to questions:
            1. Be informative, factual, and educational
            2. If you're unsure, acknowledge limitations and avoid making up facts
            3. Provide context that helps explain your answer
            4. Use a friendly, conversational tone
            5. Be concise but thorough
            """
            
            general_knowledge_messages = [{"role": "system", "content": general_knowledge_prompt}]
            
            if conversation_history and len(conversation_history) > 0:
                general_knowledge_messages.extend(conversation_history)
            
            general_knowledge_messages.append({"role": "user", "content": user_message})
            
            general_knowledge_response = client.chat.completions.create(
                model="x-ai/grok-3-mini-beta",
                messages=general_knowledge_messages
            )
            
            general_knowledge_answer = general_knowledge_response.choices[0].message.content
            
           
            
            return {
                'success': True,
                'query': user_message,
                'classification': classification,
                'results': {
                    'general_knowledge': True,
                    'interest_analysis': context_analysis if context_analysis.get("confidence", 'low') in ['medium', 'high'] else None,
                    'interest_id': interest_id
                },
                'answer': general_knowledge_answer
            }
   
        elif query_type == 'event_creation':
            print("Handling event creation")
            return smart_event_assistant.handle_smart_event_creation(
               query_type, user_message, entities, event_creation_state, session_key, user_id
            )
            
        elif query_type == 'event_scheduling':
            print("Handling event scheduling")
            return smart_scheduling_assistant.handle_smart_event_scheduling(
                query_type, user_message, entities, user_id, request
            )   
            
        else:
            # Handles other query types (like interest search and all)
            print("Handling other query types")
            
            # Find the best available dataset file(i used the generated_dataset.csv)
            if filename is None:
                possible_files = [
                    'clustered_generated_dataset.csv', 
                    'clustered_dataset.csv',
                    'generated_dataset.csv',
                    'dataset.csv',
                    'latest_clustered_data.csv'
                ]
                
                for possible_file in possible_files:
                    possible_path = os.path.join(app.config['UPLOAD_FOLDER'], possible_file)
                    if os.path.exists(possible_path):
                        filename = possible_file
                        break
                
                if filename is None:
                    return {
                        'success': False,
                        'message': 'No dataset files found. Please generate or upload a dataset first.'
                    }
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(filepath):
                return {
                    'success': False,
                    'message': f'Dataset file not found: {filename}. Please generate or upload a dataset first.'
                }
            
            # Process query with dataset
            results = {}
            
            try:
                df = pd.read_csv(filepath)
                
                # Convert string representations of lists to actual lists
                list_columns = ['interests', 'preferred_events', 'marketplace_offerings', 'marketplace_needs']
                for col in list_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x))
                
                # Handle different query types
                if query_type == 'interest_search':
                    interests = entities.get('interests', [])
                    if interests:
                        interest_results = {}
                        for interest in interests:
                            matching_users = find_users_by_interest(interest, df)
                            interest_results[interest] = matching_users
                        results['interest_results'] = interest_results
                        has_results = any(len(users) > 0 for users in interest_results.values())
                        if not has_results:
                            searched_interest = interests[0]  # Take the first interest
                            print(f"No results found for '{searched_interest}', getting smart recommendations...")
                    
                            recommendations = smart_recommender.get_smart_recommendations(searched_interest, results)
                            if recommendations and recommendations.get('similar_matches'):
                                results['smart_recommendations'] = recommendations
                                print(f"Smart recommendations found: {recommendations['similar_matches']}")
                            else:
                                print("No smart recommendations available")
                
                elif query_type == 'event_search':
                    event = entities.get('event', '')
                    if event:
                        matching_users = find_users_by_event(event, df)
                        results['event_results'] = {
                            'event': event,
                            'matching_users': matching_users
                        }
                
                        has_results = len(matching_users) > 0
        
                        if not has_results:
                            print(f"No results found for event '{event}', getting smart recommendations...")
            
                            recommendations = smart_event_recommender.get_smart_event_recommendations(event, results['event_results'])
            
                            if recommendations and recommendations.get('similar_matches'):
                                results['smart_event_recommendations'] = recommendations
                                print(f"Smart event recommendations found: {recommendations['similar_matches']}")
                            else:
                                print("No smart event recommendations available")
                
            except Exception as e:
                print(f"Error processing dataset: {str(e)}")
                return {
                    'success': False,
                    'message': f'Error processing dataset: {str(e)}'
                }

            # Generate response
            context = f"User query: {user_message}\n\n"
            if results:
                context += f"Available data: {json.dumps(results, indent=2)}\n\n"
            else:
                context += "No specific data available for this query.\n\n"
            
            messages = [{"role": "system", "content": RESPONSE_GENERATOR_PROMPT}]
            
            if conversation_history and len(conversation_history) > 0:
                messages.extend(conversation_history)
            
            messages.append({"role": "user", "content": context})
            
            final_response = client.chat.completions.create(
                model="anthropic/claude-3-opus:beta",
                messages=messages
            )
            
            answer = final_response.choices[0].message.content
            
            return {
                'success': True,
                'query': user_message,
                'classification': classification,
                'results': results,
                'answer': answer
            }
        
    except Exception as e:
        print(f"❌ ERROR in process_chat_query: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'Error: {str(e)}'}
conversation_store = {}  # Dictionary to store conversations by session ID



def get_user_filtering_state_from_database(user_id):
    """Get user's filtering state from database"""
    if not user_id:
        return None
        
    try:
        if using_mongodb:
            # MongoDB version
            user_doc = db.users.find_one({'user_id': user_id})
            
            if (user_doc and 
                user_doc.get('current_state') == 'attendee_filtering' and 
                'filtering_data' in user_doc):
                
                filtering_data = user_doc['filtering_data']
                
                # Check if state is still valid (not too old)
                state_updated = datetime.strptime(user_doc['state_updated_at'], "%Y-%m-%d %H:%M:%S")
                hours_since_update = (datetime.now() - state_updated).total_seconds() / 3600
                
                if hours_since_update < 24:  # State valid for 24 hours
                    print(f"✅ Found valid filtering state for user {user_id} from database")
                    return filtering_data
                else:
                    # Clear expired state
                    clear_user_database_state(user_id)
                    return None
                    
        else:
            # SQLite version
            conn = sqlite3.connect('messages.db')
            c = conn.cursor()
            
            c.execute('''
            SELECT current_state, state_data, updated_at 
            FROM user_states 
            WHERE user_id = ? AND current_state = 'attendee_filtering'
            ''', (user_id,))
            
            result = c.fetchone()
            conn.close()
            
            if result:
                state_data = json.loads(result[1])
                state_updated = datetime.strptime(result[2], "%Y-%m-%d %H:%M:%S")
                hours_since_update = (datetime.now() - state_updated).total_seconds() / 3600
                
                if hours_since_update < 24:  # State valid for 24 hours
                    print(f"✅ Found valid filtering state for user {user_id} from database")
                    return state_data
                else:
                    # Clear expired state
                    clear_user_database_state(user_id)
                    return None
        
        return None
        
    except Exception as e:
        print(f"Error getting user filtering state: {str(e)}")
        return None
def clear_user_database_state(user_id):
    """Clear user's state from database"""
    try:
        if using_mongodb:
            db.users.update_one(
                {'user_id': user_id},
                {'$unset': {
                    'current_state': 1,
                    'filtering_data': 1,
                    'state_updated_at': 1
                }}
            )
        else:
            conn = sqlite3.connect('messages.db')
            c = conn.cursor()
            c.execute('DELETE FROM user_states WHERE user_id = ?', (user_id,))
            conn.commit()
            conn.close()
            
        print(f"✅ Cleared database state for user {user_id}")
        
    except Exception as e:
        print(f"Error clearing user database state: {str(e)}")
def handle_response_deadline_reached(user_id, waiting_state):
    """Handle when 15-minute response period ends"""
    try:
        event_id = waiting_state['event_id']
        
        # Get response analytics
        analytics = get_event_response_analytics(event_id)
        
        if not analytics:
            return {
                'success': True,
                'answer': "I couldn't gather the response data for your event. Please try checking again later."
            }
        
        # Generate summary message
        summary = f"""🎉 **Response Collection Complete!**

Your event received {analytics['total_responded']} out of {analytics['total_invited']} responses ({analytics['response_rate']}% response rate):

✅ **Available**: {analytics['available_count']} people
❌ **Unavailable**: {analytics['unavailable_count']} people  
⏰ **Want to reschedule**: {analytics['reschedule_count']} people
❓ **Unclear/No response**: {analytics['unclear_count']} people
"""

        # Add reschedule preferences if any
        if analytics['reschedule_times']:
            summary += "\n**Reschedule preferences:**\n"
            for time_pref, count in sorted(analytics['reschedule_times'].items(), key=lambda x: x[1], reverse=True):
                summary += f"• {time_pref}: {count} {'person' if count == 1 else 'people'}\n"
        
        summary += f"\n💡 You can now ask follow-up questions to filter your {analytics['available_count']} available attendees!"
        summary += "\nFor example: 'How many people have soccer cleats?' or 'Who can play goalkeeper?'"
        
        # Clear waiting state and set filtering state
        waiting_key = f"waiting_for_responses_{user_id}"
        filtering_key = f"attendee_filtering_{user_id}"
        
        session.pop(waiting_key, None)
        session[filtering_key] = {
            'event_id': event_id,
            'available_attendees': get_available_attendee_ids(event_id),
            'total_available': analytics['available_count'],
            'applied_filters': [],
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        session.modified = True
        
        return {
            'success': True,
            'answer': summary
        }
        
    except Exception as e:
        print(f"Error handling response deadline: {str(e)}")
        return {
            'success': True,
            'answer': "Your event response period has ended. You can now start filtering your attendees by asking questions like 'How many people have soccer cleats?'"
        }

def get_event_response_analytics(event_id):
    """Get analytics about responses to an event invitation"""
    try:
        if using_mongodb:
            # MongoDB version
            # Get all responses for this event
            responses = list(db.event_responses.find({'event_id': event_id}))
            
            # Get all invitations sent for this event to calculate total invited
            invitations = list(db.messages.find({
                'event_id': event_id,
                'message_type': 'event_invitation'
            }))
            
        else:
            # SQLite version
            conn_responses = sqlite3.connect('social_platform.db')
            c_responses = conn_responses.cursor()
            
            # Get all responses for this event
            c_responses.execute('SELECT * FROM event_responses WHERE event_id = ?', (event_id,))
            response_rows = c_responses.fetchall()
            
            # Convert to list of dictionaries for consistency
            responses = []
            for row in response_rows:
                responses.append({
                    'user_id': row[2],  # user_id column
                    'status': row[3],   # status column
                    'time': row[4],     # time column
                    'response_text': row[5],  # response_text column
                    'updated_at': row[6]      # updated_at column
                })
            
            conn_responses.close()
            
            # Get all invitations sent for this event
            conn_messages = sqlite3.connect('messages.db')
            c_messages = conn_messages.cursor()
            
            c_messages.execute('''
                SELECT to_user_id FROM messages 
                WHERE event_id = ? AND message_type = 'event_invitation'
            ''', (event_id,))
            
            invitation_rows = c_messages.fetchall()
            invitations = [{'to_user_id': row[0]} for row in invitation_rows]
            
            conn_messages.close()
        
        # Calculate analytics
        total_invited = len(invitations)
        total_responded = len(responses)
        
        # Count by status
        available_count = len([r for r in responses if r['status'] == 'AVAILABLE'])
        unavailable_count = len([r for r in responses if r['status'] == 'UNAVAILABLE'])
        reschedule_count = len([r for r in responses if r['status'] == 'RESCHEDULE'])
        unclear_count = len([r for r in responses if r['status'] == 'UNCLEAR'])
        
        # Calculate response rate
        response_rate = round((total_responded / total_invited * 100), 1) if total_invited > 0 else 0
        
        # Collect reschedule time preferences
        reschedule_times = {}
        for response in responses:
            if response['status'] == 'RESCHEDULE' and response['time'] and response['time'] != 'NONE':
                time_pref = response['time']
                reschedule_times[time_pref] = reschedule_times.get(time_pref, 0) + 1
        
        # Get user IDs who are available (for filtering feature)
        available_user_ids = [r['user_id'] for r in responses if r['status'] == 'AVAILABLE']
        
        analytics = {
            'event_id': event_id,
            'total_invited': total_invited,
            'total_responded': total_responded,
            'response_rate': response_rate,
            'available_count': available_count,
            'unavailable_count': unavailable_count,
            'reschedule_count': reschedule_count,
            'unclear_count': unclear_count,
            'reschedule_times': reschedule_times,
            'available_user_ids': available_user_ids,
            'no_response_count': total_invited - total_responded
        }
        
        print(f"Event {event_id} analytics: {analytics}")
        return analytics
        
    except Exception as e:
        print(f"Error getting event response analytics: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


def get_available_attendee_ids(event_id):
    """Get list of user IDs who are available for the event"""
    try:
        if using_mongodb:
            responses = list(db.event_responses.find({
                'event_id': event_id,
                'status': 'AVAILABLE'
            }))
            return [resp['user_id'] for resp in responses]
        else:
            conn = sqlite3.connect('social_platform.db')
            c = conn.cursor()
            c.execute('SELECT user_id FROM event_responses WHERE event_id = ? AND status = ?', 
                     (event_id, 'AVAILABLE'))
            result = [row[0] for row in c.fetchall()]
            conn.close()
            return result
    except Exception as e:
        print(f"Error getting available attendees: {str(e)}")
        return []


  # Dictionary to store conversations by session ID

def ensure_tables_exist():
    """Make sure required tables exist"""
    if using_mongodb:
        # MongoDB creates collections automatically, no need to do anything else
        pass
    else:
        # Original SQLite code
        create_event_tables()
        init_db()
# Add these database and management functions

def create_event_tables():
    """Initialize database tables for events and responses"""
    conn = sqlite3.connect('social_platform.db')
    c = conn.cursor()
    
    # Create events table
    c.execute('''
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        creator_id INTEGER,
        title TEXT,
        date TEXT,
        time TEXT,
        location TEXT,
        description TEXT,
        created_at TEXT,
        event_state TEXT,
        last_updated TEXT
        deadline_notification_sent INTEGER DEFAULT 0  
    )
    ''')
    
    # Create event responses table 
    c.execute('''
    CREATE TABLE IF NOT EXISTS event_responses (
        id INTEGER PRIMARY KEY,
        event_id TEXT,
        user_id INTEGER,
        status TEXT,
        time TEXT,
        response_text TEXT,
        updated_at TEXT,
        FOREIGN KEY (event_id) REFERENCES events (event_id),
        UNIQUE(event_id, user_id)
    )
    ''')
    
    conn.commit()
    conn.close()



def save_event(event_details):
    """Save a new event to the database - UPDATED"""
    event_id = event_details.get('event_id', f"event_{datetime.now().strftime('%Y%m%d%H%M%S')}_{event_details.get('creator_id', 0)}")
    
    creator_id = event_details.get('creator_id')
    if creator_id is None:
        creator_id = 0
        print("Warning: Creator ID is None, defaulting to 0")
        event_details['creator_id'] = creator_id
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if using_mongodb:
            event_doc = {
                'event_id': event_id,
                'creator_id': creator_id,
                'title': event_details.get('title'),
                'date': event_details.get('date'),
                'time': event_details.get('time'),
                'location': event_details.get('location'),
                'description': event_details.get('description', f"A {event_details.get('title')} event"),
                'created_at': current_time,
                'event_state': event_details,
                'last_updated': current_time,
                'participants': [],
                'deadline_notification_sent': False  # Track if notification was sent
            }
            
            db.events.insert_one(event_doc)
        else:
            conn = sqlite3.connect('social_platform.db')
            c = conn.cursor()
            
            c.execute('''
            INSERT INTO events 
            (event_id, creator_id, title, date, time, location, description, created_at, event_state, last_updated) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, creator_id, 
                event_details.get('title'), 
                event_details.get('date'), 
                event_details.get('time'), 
                event_details.get('location'), 
                event_details.get('description', f"A {event_details.get('title')} event"), 
                current_time,
                json.dumps(event_details),
                current_time
            ))
            
            conn.commit()
            conn.close()
        
        print(f"Event {event_id} saved successfully at {current_time}")
        return event_id
    except Exception as e:
        print(f"Error saving event: {e}")
        return None
    
def save_event_response(event_id, user_id, status, time, response_text):
    """Save a user's response to an event invitation"""
    try:
        # Ensures that we have valid values
        if event_id is None or user_id is None:
            print(f"Warning: Missing required data for event response: event_id={event_id}, user_id={user_id}")
            return False
            
        # Format values for database
        status = status or "UNCLEAR"
        time = time or "NONE"
        response_text = response_text or ""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if using_mongodb:
            # MongoDB version
            response_doc = {
                'user_id': user_id,
                'status': status,
                'time': time,
                'response_text': response_text,
                'updated_at': current_time
            }
            
            # Update or add participant to the event document
            result = db.events.update_one(
                {'event_id': event_id, 'participants.user_id': user_id},
                {'$set': {'participants.$': response_doc}}
            )
            
            # If the user is not already a participant add the user 
            if result.matched_count == 0:
                db.events.update_one(
                    {'event_id': event_id},
                    {'$push': {'participants': response_doc}}
                )
                
            # create a separate event_responses document
            db.event_responses.update_one(
                {'event_id': event_id, 'user_id': user_id},
                {'$set': {
                    'event_id': event_id,
                    'user_id': user_id,
                    'status': status,
                    'time': time,
                    'response_text': response_text,
                    'updated_at': current_time
                }},
                upsert=True
            )
        else:
            # Original SQLite code
            conn = sqlite3.connect('social_platform.db')
            c = conn.cursor()
            
            # Insert or update the response
            c.execute('''
            INSERT OR REPLACE INTO event_responses 
            (event_id, user_id, status, time, response_text, updated_at) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event_id, 
                user_id, 
                status, 
                time, 
                response_text, 
                current_time
            ))
            
            conn.commit()
            conn.close()
        
        print(f"Successfully saved response for event {event_id} from user {user_id}")
        return True
        
    except Exception as e:
        print(f"Error saving event response: {str(e)}")
        return False
    
def get_recent_event_invitation(user_id, timeframe_minutes=30):
    """Get the most recent event invitation for a user within the specified timeframe"""
    try:
        if not using_mongodb:
            return None
            
        # Calculate the cutoff time (current time minus timeframe)
        cutoff_time = datetime.now() - timedelta(minutes=timeframe_minutes)
        cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Find the most recent event invitation in MongoDB
        recent_invitation = db.messages.find_one({
            'to_user_id': user_id,
            'message_type': 'event_invitation',
            'is_read': 0,  # Still unread
            'timestamp': {'$gte': cutoff_str}  # Within timeframe
        }, sort=[('timestamp', -1)])  # Sort by most recent first
        
        return recent_invitation
    except Exception as e:
        print(f"Error getting recent event invitation: {e}")
        return None
    
def get_event_details(event_id):
    """Retrieve event details from the database based on event ID"""
    try:
        if using_mongodb:
            # MongoDB version
            event_doc = db.events.find_one({'event_id': event_id})
            
            if not event_doc:
                return None
            
            # Convert ObjectId to string for JSON serialization
            if '_id' in event_doc:
                event_doc['_id'] = str(event_doc['_id'])
            
            # Extract the event details
            event_details = {
                'event_id': event_doc['event_id'],
                'creator_id': event_doc['creator_id'],
                'title': event_doc['title'],
                'date': event_doc['date'],
                'time': event_doc['time'],
                'location': event_doc['location'],
                'description': event_doc['description']
            }
            
            # If event state is stored, merge it
            if 'event_state' in event_doc and event_doc['event_state']:
                for key, value in event_doc['event_state'].items():
                    if key not in event_details:  # Don't overwrite primary fields
                        event_details[key] = value
            
            return event_details
        else:
            # Original SQLite code
            conn = sqlite3.connect('social_platform.db')
            c = conn.cursor()
            
            # Query the events table for the specified event
            c.execute('''
            SELECT event_id, creator_id, title, date, time, location, description, event_state
            FROM events 
            WHERE event_id = ?
            ''', (event_id,))
            
            result = c.fetchone()
            conn.close()
            
            if not result:
                return None
            
            # Construct the event details dictionary
            event_details = {
                'event_id': result[0],
                'creator_id': result[1],
                'title': result[2],
                'date': result[3],
                'time': result[4],
                'location': result[5],
                'description': result[6]
            }
            
            # If event state is stored as JSON, parse it
            if result[7]:
                try:
                    state_data = json.loads(result[7])
                    # Update with additional state information
                    for key, value in state_data.items():
                        if key not in event_details:  # Don't overwrite primary fields
                            event_details[key] = value
                except json.JSONDecodeError:
                    pass  # If we can't parse the JSON, just use the basic details
            
            return event_details
            
    except Exception as e:
        print(f"Error retrieving event details: {str(e)}")
        return None
    
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat requests with improved session management
    """
    # Extract request parameters
    user_message = request.json.get('user_message', '')
    event_message_id = request.json.get('event_message_id')
    event_response = request.json.get('event_response', False)
    user_id = session.get('current_user_id') or request.json.get('user_id')
    
    # IMPORTANT: Ensure user_id is an integer
    if user_id and not isinstance(user_id, int):
        try:
            user_id = int(user_id)
        except ValueError:
            print(f"[WARNING] Invalid user_id format: {user_id}")
            user_id = None
    
    # Log request details
    print(f"[DEBUG] Request to /api/chat, method=POST, current_user_id={user_id}")
    print(f"[DEBUG] Chat request: user_message='{user_message}', event_message_id={event_message_id}, event_response={event_response}")
    print(f"[DEBUG] Session state: {session}")
    
    # Debug log session state before processing
    log_session_state(user_id, "BEFORE")
    
    # Check for legacy session format and migrate if needed
    if 'event_creation_state' in session and user_id and f"event_creation_state_{user_id}" not in session:
        print(f"[DEBUG] Auto-migrating legacy session for user {user_id}")
        session[f"event_creation_state_{user_id}"] = session['event_creation_state'].copy()
        
        # Update creator_id to match current user
        if 'event_details' in session[f"event_creation_state_{user_id}"]:
            session[f"event_creation_state_{user_id}"]['event_details']['creator_id'] = user_id
            print(f"[DEBUG] Updated creator_id to {user_id} in migrated session")
        
        # Remove old session format after migration
        session.pop('event_creation_state', None)
        session.modified = True
        print(f"[DEBUG] Migration complete, new session state: {session}")
    
    # Handle conversation history
    session_id = request.json.get('session_id')
    if not session_id:
        # Generate a new session ID if none provided
        import time
        import random
        import string
        def generate_id(length=10):
            return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))
        
        session_id = f"session_{int(time.time()*1000)}_{generate_id()}"
        print(f"[DEBUG] Created new session_id={session_id}")
    else:
        print(f"[DEBUG] Session ID: {session_id}")
    
    # Initialize or retrieve conversation history
    if session_id not in conversation_store:
        print(f"[DEBUG] Creating new conversation history for session_id={session_id}")
        conversation_store[session_id] = []
    
    conversation_history = conversation_store[session_id]
    print(f"[DEBUG] Conversation history length: {len(conversation_history)}")
    
    # Find appropriate dataset file to use
    filename = request.json.get('filename')
    if filename and not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        print(f"[DEBUG] Requested file '{filename}' not found, searching for alternatives...")
        filename = None
    if not filename:
        for possible_file in ['clustered_dataset.csv', 'dataset.csv', 'generated_dataset.csv']:
            possible_path = os.path.join(app.config['UPLOAD_FOLDER'], possible_file)
            if os.path.exists(possible_path):
                filename = possible_file
                print(f"[DEBUG] Using file: {filename}")
                break
    
    # Process the chat query
    print(f"[DEBUG] Calling process_chat_query with user_message='{user_message}', filename={filename}")
    result = process_chat_query(
        user_message=user_message,
        filename=filename,
        conversation_history=conversation_history,
        user_id=user_id,
        request=request
    )
    
    # Log results and session state after processing
    print(f"[DEBUG] process_chat_query result: success={result.get('success', False)}")
    log_session_state(user_id, "AFTER")
    
    # Handle unsuccessful query processing
    if not result.get('success', False):
        error_message = result.get('message', 'An error occurred processing your request')
        print(f"[ERROR] Chat processing failed: {error_message}")
        return jsonify({
            'success': False,
            'message': error_message,
            'session_id': session_id
        })
    
    # Get the response text
    answer = result.get('answer', '')
    
    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": answer})
    
    # Trim conversation history if it's getting too long
    max_history = 20  # Maximum number of messages to keep
    if len(conversation_history) > max_history * 2:  # Each exchange is 2 messages
        # Keep the most recent exchanges
        conversation_history = conversation_history[-max_history*2:]
        conversation_store[session_id] = conversation_history
        print(f"[DEBUG] Trimmed conversation history to {len(conversation_history)} messages")
    
    # Prepare response
    print(f"[DEBUG] Returning response with session_id={session_id}")
    return jsonify({
        'success': True,
        'message': answer,
        'session_id': session_id,
        'query_classification': result.get('classification', {}),
        'user_id': user_id
    })


def log_session_state(user_id, prefix=""):
    """Log current session state for debugging"""
    try:
        user_specific_key = f"event_creation_state_{user_id}" if user_id else "event_creation_state"
        legacy_data = session.get('event_creation_state', {})
        user_specific_data = session.get(user_specific_key, {})
        
        print(f"{prefix} SESSION DEBUG - user_id: {user_id}")
        print(f"{prefix} SESSION DEBUG - all keys: {list(session.keys())}")
        print(f"{prefix} SESSION DEBUG - legacy data: {legacy_data}")
        print(f"{prefix} SESSION DEBUG - user-specific data: {user_specific_data}")
        
        # Check for mismatch in creator_id
        if (legacy_data and user_specific_data and 
            'event_details' in legacy_data and 'event_details' in user_specific_data and
            legacy_data['event_details'].get('creator_id') != user_specific_data['event_details'].get('creator_id')):
            print(f"{prefix} SESSION DEBUG - WARNING: creator_id mismatch between legacy and user-specific")
    except Exception as e:
        print(f"{prefix} SESSION DEBUG - error logging session: {str(e)}")

# 2. Add session management to your upload route
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = f"dataset_{session_id}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': filename,
            'session_id': session_id
        })
    
    return jsonify({'success': False, 'error': 'Invalid file type'})


@app.route('/api/reset_database', methods=['POST'])
def reset_database():
    """Reset the event and message databases or clear old data"""
    try:
        if using_mongodb:
            # MongoDB version - clear all collections
            db.messages.delete_many({})
            db.event_responses.delete_many({})
            db.events.delete_many({})
            print("MongoDB database reset successful")
        else:
            # SQLite version
            # Connect to messages database
            conn_messages = sqlite3.connect('messages.db')
            c_messages = conn_messages.cursor()
            c_messages.execute('DELETE FROM messages')
            conn_messages.commit()
            conn_messages.close()
            
            # Connect to social platform database
            conn_events = sqlite3.connect('social_platform.db')
            c_events = conn_events.cursor()
            c_events.execute('DELETE FROM events')
            c_events.execute('DELETE FROM event_responses')
            conn_events.commit()
            conn_events.close()
            
            print("SQLite database reset successful")
        
        return True
    except Exception as e:
        print(f"Error resetting database: {str(e)}")
        return False
    
@app.route('/api/reset_all', methods=['POST'])
def reset_all():
    """Reset everything - database and session data"""
    # Reset database
    reset_database()
    
    # Clear all sessions
    session.clear()
    session.modified = True
    
    # Reset global variable
    global current_user_id
    current_user_id = None
    
    # Also clear conversation store
    conversation_store.clear()
    
    return jsonify({
        'success': True,
        'message': 'All data has been reset successfully'
    })           

    
# 4. Add multiple interest search with fuzzy matching
@app.route('/api/interests/multiple/<interests>/users', methods=['GET'])
def find_users_by_multiple_interests(interests):
    """Find users with multiple interests (any of the specified interests)"""
    try:
        # Parse the comma-separated interests
        interest_list = interests.split(',')
        filename = request.args.get('filename', 'generated_dataset.csv')
        
        # Find an existing file if the default doesn't exist
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            # Try to find any clustered or generated dataset
            possible_files = [
                'clustered_generated_dataset.csv',
                'clustered_dataset.csv',
                'generated_dataset.csv',
                'dataset.csv'
            ]
            
            for possible_file in possible_files:
                possible_path = os.path.join(app.config['UPLOAD_FOLDER'], possible_file)
                if os.path.exists(possible_path):
                    filepath = possible_path
                    break
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'No dataset available'})
        
        print(f"Using dataset: {filepath} to find users with interests: {interest_list}")
        
        # Read the dataset
        df = pd.read_csv(filepath)
        
        # Convert interests from string to list if needed
        if isinstance(df['interests'].iloc[0], str):
            df['interests'] = df['interests'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Normalize the search interests (lowercase for comparison)
        normalized_interests = [interest.lower() for interest in interest_list]
        
        # Track users with each interest
        matching_users = []
        interest_matches = {interest: [] for interest in interest_list}
        
        for _, row in df.iterrows():
            # Convert each user's interests to lowercase for comparison
            user_interests = [i.lower() for i in row['interests']]
            
            # Check which searched interests match this user
            matched_interests = []
            
            for norm_interest in normalized_interests:
                exact_match = False
                fuzzy_match = False
                matched_interest = ""
                
                # Check for exact match
                if norm_interest in user_interests:
                    exact_match = True
                    matched_interest = norm_interest
                
                # If no exact match, try fuzzy matching
                if not exact_match:
                    for user_interest in user_interests:
                        # Option 1: One is contained in the other
                        if norm_interest in user_interest or user_interest in norm_interest:
                            fuzzy_match = True
                            matched_interest = user_interest
                            break
                        
                        # Option 2: Levenshtein distance (for typos)
                        # Only check if the lengths are reasonably close
                        if abs(len(norm_interest) - len(user_interest)) <= 3:
                            # Calculate edit distance
                            distance = levenshtein_distance(norm_interest, user_interest)
                            # If distance is small relative to length, consider it a match
                            if distance <= 2 or distance <= len(norm_interest) * 0.3:
                                fuzzy_match = True
                                matched_interest = user_interest
                                break
                
                if exact_match or fuzzy_match:
                    # Record which original interest this matched
                    original_interest = interest_list[normalized_interests.index(norm_interest)]
                    matched_interests.append({
                        'search_interest': original_interest,
                        'matched_interest': matched_interest,
                        'exact_match': exact_match
                    })
                    
                    # Add to the interest-specific match list
                    interest_matches[original_interest].append(int(row['user_id']))
            
            # If this user matches any of the interests, add them
            if matched_interests:
                user_info = {
                    'user_id': int(row['user_id']),
                    'age': int(row['age']),
                    'location': row['location'],
                    'matched_interests': matched_interests,
                    'interest_count': len(matched_interests)  # Track how many interests matched
                }
                
                if 'cluster' in df.columns:
                    user_info['cluster'] = int(row['cluster'])
                
                matching_users.append(user_info)
        
        # For each interest, count how many users have it
        interest_counts = {interest: len(users) for interest, users in interest_matches.items()}
        
        # Calculate counts for users with combinations of interests (pairwise combinations)
        pair_combination_counts = {}
        if len(interest_list) >= 2:
            for i in range(len(interest_list)):
                for j in range(i+1, len(interest_list)):
                    interest1 = interest_list[i]
                    interest2 = interest_list[j]
                    
                    # Find users who match both interests
                    common_users = set(interest_matches[interest1]) & set(interest_matches[interest2])
                    
                    pair_combination_counts[f"{interest1}, {interest2}"] = len(common_users)
        
        # Calculate counts for ALL combinations (including 3+)
        combination_counts = pair_combination_counts.copy()
        
        # For 3+ interest combinations
        if len(interest_list) >= 3:
            # For all possible combinations of 3 or more
            for r in range(3, len(interest_list) + 1):
                for combo in itertools.combinations(interest_list, r):
                    # Start with all users for first interest
                    if not interest_matches[combo[0]]:
                        common_users = set()
                    else:
                        common_users = set(interest_matches[combo[0]])
                    
                    # Intersect with users for all other interests in combo
                    for interest in combo[1:]:
                        common_users = common_users & set(interest_matches[interest])
                    
                    # Add to combination counts
                    combo_key = ", ".join(combo)
                    combination_counts[combo_key] = len(common_users)
        
        # Add a special category for users matching ALL interests
        all_interests_key = ", ".join(interest_list)
        if len(interest_list) >= 2:
            # Find users who match ALL searched interests
            all_users = None
            for interest in interest_list:
                users = set(interest_matches[interest])
                if all_users is None:
                    all_users = users
                else:
                    all_users = all_users & users
            
            combination_counts["all_interests"] = len(all_users) if all_users else 0
            
            # Mark users who match ALL interests
            if all_users:
                for user in matching_users:
                    if user['user_id'] in all_users:
                        user['matches_all'] = True
                    else:
                        user['matches_all'] = False
        
        # Sort users by number of matched interests (most matches first)
        matching_users.sort(key=lambda x: x['interest_count'], reverse=True)
        
        # For debugging
        print(f"Found {len(matching_users)} users matching at least one interest")
        print(f"Interest counts: {interest_counts}")
        print(f"Combination counts: {combination_counts}")
        if "all_interests" in combination_counts:
            print(f"Users matching ALL interests: {combination_counts['all_interests']}")
        
        return jsonify({
            'success': True,
            'interests': interest_list,
            'count': len(matching_users),
            'users': matching_users,
            'interest_counts': interest_counts,
            'combination_counts': combination_counts
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/chat')
def chat_interface():
    return render_template('chat.html')

# Add these imports at the top if not already present
# Add this at the top level of your file
current_user_id = None

@app.route('/get_users')
def get_users():
    """Get a list of users from the dataset for the dropdown"""
    filename = request.args.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'message': 'No filename provided'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'File not found'})
    
    try:
        df = pd.read_csv(filepath)
        users = []
        
        # Get all users instead of just a sample
        for _, row in df.iterrows():
            users.append({
                'user_id': int(row['user_id']),
                'age': int(row['age']),
                'location': row['location']
            })
        
        return jsonify({
            'success': True,
            'users': users
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/set_user_context', methods=['POST'])
def set_user_context():
    """Set the current user context using a global variable"""
    global current_user_id
    data = request.get_json()
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'No user ID provided'})
    
    # Store in global variable instead of session
    current_user_id = int(user_id)
    
    return jsonify({'success': True})

@app.route('/get_user_context')
def get_user_context():
    """Get the current user context from the global variable"""
    global current_user_id
    user_id = current_user_id
    
    print(f"[DEBUG] Current user context: user_id={user_id}")
    print(f"[DEBUG] Session contents: {session}")  # Print entire session
    
    if not user_id:
        return jsonify({'success': False, 'message': 'No user selected'})
    
    # Get the user details from the dataset
    # Find a suitable dataset file
    available_files = os.listdir(app.config['UPLOAD_FOLDER'])
    csv_files = [f for f in available_files if f.endswith('.csv')]
    
    if not csv_files:
        return jsonify({'success': True, 'user_id': user_id, 'details': None})
    
    # Prefer clustered files
    clustered_files = [f for f in csv_files if f.startswith('clustered_')]
    filename = clustered_files[0] if clustered_files else csv_files[0]
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        df = pd.read_csv(filepath)
        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            return jsonify({'success': True, 'user_id': user_id, 'details': None})
        
        # Convert string lists to actual lists
        user_row = user_data.iloc[0].copy()
        list_columns = ['interests', 'preferred_events', 'marketplace_offerings', 'marketplace_needs']
        
        for col in list_columns:
            if col in user_row and isinstance(user_row[col], str):
                user_row[col] = eval(user_row[col])
        
        # Create user details
        user_details = {
            'age': int(user_row['age']),
            'location': user_row['location'],
            'interests': user_row['interests'],
            'preferred_events': user_row['preferred_events'],
            'cluster': int(user_row['cluster']) if 'cluster' in user_row else None
        }
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'details': user_details
        })
    
    except Exception as e:
        return jsonify({'success': True, 'user_id': user_id, 'details': None, 'error': str(e)})

def init_db():
    """Initialize the database with message tables"""
    if using_mongodb:
        # MongoDB collections are already created in the connection block
        pass
    else:
        # Original SQLite code
        conn = sqlite3.connect('messages.db')
        c = conn.cursor()
        
        # Create messages table
        c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            from_user_id INTEGER,
            to_user_id INTEGER,
            message_type TEXT,
            content TEXT,
            event_id TEXT,
            event_details TEXT,
            timestamp TEXT,
            is_read INTEGER DEFAULT 0,
            response TEXT DEFAULT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
# Call this function when the app starts
init_db()

# Add a route to handle event creation
# Replace your @app.route('/api/get_messages') with this version:

# Replace your entire @app.route('/api/get_messages') function with this:

@app.route('/api/get_messages')
def get_messages():
    """Get messages for the current user - FIXED to show analytics"""
    global current_user_id
    user_id = current_user_id
    
    print(f"[DEBUG] Loading messages for user_id={user_id}")
    
    if not user_id:
        print("[DEBUG] No user logged in, returning error")
        return jsonify({'success': False, 'message': 'No user logged in'})
    
    # Helper function to clean MongoDB ObjectIds
    def clean_mongo_doc(doc):
        """Convert MongoDB ObjectId to string for JSON serialization"""
        if doc and isinstance(doc, dict):
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        return doc
    
    def clean_mongo_list(docs):
        """Clean a list of MongoDB documents"""
        return [clean_mongo_doc(doc) for doc in docs]
    
    try:
        if using_mongodb:
            # Get received messages
            received_cursor = db.messages.find({'to_user_id': user_id}).sort('timestamp', 1)
            received = []
            for doc in received_cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                received.append(doc)
            
            # Get sent messages
            sent_cursor = db.messages.find({'from_user_id': user_id}).sort('timestamp', 1)
            sent = []
            for doc in sent_cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                sent.append(doc)
            
            # Count unread messages
            unread_count = db.messages.count_documents({'to_user_id': user_id, 'is_read': 0})
            
            # CHECK FOR ANALYTICS MESSAGES (with proper ObjectId handling)
            analytics_message = db.messages.find_one({
                'to_user_id': user_id,
                'message_type': 'event_analytics'
            }, sort=[('timestamp', -1)])
            
            # Clean the analytics message
            analytics_message = clean_mongo_doc(analytics_message)
            
        else:
            # SQLite version
            conn = sqlite3.connect('messages.db')
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get received messages
            c.execute('''
            SELECT * FROM messages 
            WHERE to_user_id = ? 
            ORDER BY timestamp DESC
            ''', (user_id,))
            
            received = []
            for row in c.fetchall():
                message = dict(row)
                if message['event_details']:
                    try:
                        message['event_details'] = json.loads(message['event_details'])
                    except json.JSONDecodeError:
                        message['event_details'] = {}
                received.append(message)
            
            # Get sent messages
            c.execute('''
            SELECT * FROM messages 
            WHERE from_user_id = ? 
            ORDER BY timestamp DESC
            ''', (user_id,))
            
            sent = []
            for row in c.fetchall():
                message = dict(row)
                if message['event_details']:
                    try:
                        message['event_details'] = json.loads(message['event_details'])
                    except json.JSONDecodeError:
                        message['event_details'] = {}
                sent.append(message)
            
            # Count unread messages
            c.execute('SELECT COUNT(*) FROM messages WHERE to_user_id = ? AND is_read = 0', (user_id,))
            unread_count = c.fetchone()[0]
            
            # CHECK FOR ANALYTICS MESSAGES
            c.execute('''
            SELECT * FROM messages 
            WHERE to_user_id = ? AND message_type = 'event_analytics'
            ORDER BY timestamp DESC LIMIT 1
            ''', (user_id,))
            
            analytics_row = c.fetchone()
            analytics_message = dict(analytics_row) if analytics_row else None
            
            conn.close()
        
        # PRIORITY CHECK: If user has analytics message, show that instead
        if analytics_message:
            print(f"[DEBUG] Found analytics message for user {user_id}")
            
            # Additional safety check for ObjectId conversion
            if isinstance(analytics_message, dict) and '_id' in analytics_message:
                analytics_message['_id'] = str(analytics_message['_id'])
            
            # Also clean any nested ObjectIds in analytics_data
            if 'analytics_data' in analytics_message and isinstance(analytics_message['analytics_data'], dict):
                analytics_message['analytics_data'] = clean_mongo_doc(analytics_message['analytics_data'])
            
            return jsonify({
                'success': True,
                'received': received,
                'sent': sent,
                'unread_count': unread_count,
                'analytics_mode': True,
                'analytics_message': analytics_message
            })
        
        # CHECK: If user is in waiting mode, show waiting status
        waiting_key = f"waiting_for_responses_{user_id}"
        from flask import session
        if waiting_key in session:
            waiting_state = session[waiting_key]
            deadline = datetime.strptime(waiting_state['deadline'], "%Y-%m-%d %H:%M:%S")
            
            if datetime.now() > deadline:
                print(f"[DEBUG] User {user_id} deadline expired but no analytics message found")
                # Force generate analytics now
                try:
                    result = handle_response_deadline_reached(user_id, waiting_state)
                    return jsonify({
                        'success': True,
                        'received': received,
                        'sent': sent,
                        'unread_count': unread_count,
                        'analytics_mode': True,
                        'analytics_generated': True,
                        'analytics_content': result.get('answer', 'Analytics generated')
                    })
                except Exception as e:
                    print(f"[DEBUG] Error generating analytics: {e}")
            else:
                # Still in waiting period - show countdown
                remaining_time = deadline - datetime.now()
                minutes_left = int(remaining_time.total_seconds() / 60)
                seconds_left = int(remaining_time.total_seconds() % 60)
                
                # Get current response count if possible
                event_id = waiting_state.get('event_id')
                current_responses = 0
                
                try:
                    if using_mongodb and event_id:
                        current_responses = db.event_responses.count_documents({'event_id': event_id})
                    elif event_id:
                        conn = sqlite3.connect('social_platform.db')
                        c = conn.cursor()
                        c.execute('SELECT COUNT(*) FROM event_responses WHERE event_id = ?', (event_id,))
                        result = c.fetchone()
                        current_responses = result[0] if result else 0
                        conn.close()
                except Exception as e:
                    print(f"Error getting response count: {e}")
                
                response_rate = (current_responses / waiting_state.get('invitations_sent', 1) * 100) if waiting_state.get('invitations_sent', 0) > 0 else 0
                
                waiting_message = f"""⏳ **Response Collection in Progress**

📅 **Event:** {waiting_state.get('event_details', {}).get('title', 'Your Event')}
📧 **Invitations Sent:** {waiting_state.get('invitations_sent', 0)} users
📬 **Responses So Far:** {current_responses} received ({response_rate:.1f}% response rate)

⏰ **Time Remaining:** {minutes_left} minutes, {seconds_left} seconds

🔄 **What's happening:**
• Users are receiving and responding to your event invitation
• I'm collecting all responses in real-time
• You'll get complete analytics automatically when the 15-minute period ends

✨ **Just sit back and relax - I'll handle everything!**"""

                return jsonify({
                    'success': True,
                    'received': received,
                    'sent': sent,
                    'unread_count': unread_count,
                    'waiting_mode': True,
                    'waiting_message': waiting_message,
                    'time_remaining': f"{minutes_left}:{seconds_left:02d}",
                    'responses_received': current_responses,
                    'response_rate': round(response_rate, 1)
                })
        
        # Default: Normal message loading
        return jsonify({
            'success': True,
            'received': received,
            'sent': sent,
            'unread_count': unread_count
        })
        
    except Exception as e:
        print(f"[DEBUG] Error getting messages: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})
        
@app.route('/debug/check_data', methods=['GET'])

@app.route('/api/reset_sessions', methods=['POST', 'GET'])
def reset_sessions():
    """Clear all session data or migrate to user-specific format"""
    action = request.args.get('action', 'reset')
    user_id = request.args.get('user_id')
    
    if action == 'reset':
        # Clear all session data
        session.clear()
        message = "All sessions cleared!"
    elif action == 'migrate' and user_id:
        # Migrate legacy session to user-specific format
        user_id = int(user_id)
        if 'event_creation_state' in session:
            session_key = f"event_creation_state_{user_id}"
            # Migrate data
            session[session_key] = session['event_creation_state'].copy()
            # Update creator_id
            if 'event_details' in session[session_key]:
                session[session_key]['event_details']['creator_id'] = user_id
            # Remove old session
            session.pop('event_creation_state', None)
            message = f"Session migrated for user {user_id}"
        else:
            message = "No session data to migrate"
    else:
        message = "Invalid action"
    
    session.modified = True
    
    # Return all session data for debugging
    session_info = {k: str(v)[:100] + '...' if isinstance(v, (dict, list)) and len(str(v)) > 100 else v 
                    for k, v in session.items()}
    
    return {
        'success': True, 
        'message': message,
        'session_keys': list(session.keys()),
        'session_info': session_info
    }
def debug_check_data():
    """Check if event data exists in the database"""
    # Check messages table for event invitations
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute('SELECT id, from_user_id, to_user_id, message_type, event_id FROM messages WHERE message_type = "event_invitation"')
    invitations = c.fetchall()
    
    # Check events table for events
    conn2 = sqlite3.connect('social_platform.db')
    c2 = conn2.cursor()
    c2.execute('SELECT event_id, creator_id, title, date, time FROM events')
    events = c2.fetchall()
    
    # Close connections
    conn.close()
    conn2.close()
    
    # Print to terminal and return results
    print(f"[DEBUG] Found {len(invitations)} event invitations:")
    for inv in invitations:
        print(f"  - ID: {inv[0]}, From: {inv[1]}, To: {inv[2]}, Event: {inv[4]}")
    
    print(f"[DEBUG] Found {len(events)} events:")
    for evt in events:
        print(f"  - ID: {evt[0]}, Creator: {evt[1]}, Title: {evt[2]}")
    
    return jsonify({
        'invitations': [{'id': i[0], 'from': i[1], 'to': i[2], 'event_id': i[4]} for i in invitations],
        'events': [{'id': e[0], 'creator': e[1], 'title': e[2]} for e in events]
    })

@app.before_request
def log_request_info():
    """Log details about each request and global state"""
    global current_user_id
    print(f"[DEBUG] Request to {request.path}, method={request.method}, current_user_id={current_user_id}")
    
@app.route('/debug/check_messages', methods=['GET'])
def debug_check_messages():
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute('SELECT * FROM messages LIMIT 10')
    messages = c.fetchall()
    conn.close()
    
    conn2 = sqlite3.connect('social_platform.db')
    c2 = conn2.cursor()
    c2.execute('SELECT * FROM events LIMIT 10')
    events = c2.fetchall()
    c2.execute('SELECT * FROM event_responses LIMIT 10')
    responses = c2.fetchall()
    conn2.close()
    
    # Print to terminal for debugging
    print("\n===== MESSAGES =====")
    for msg in messages:
        print(msg)
    
    print("\n===== EVENTS =====")
    for event in events:
        print(event)
    
    print("\n===== EVENT RESPONSES =====")
    for resp in responses:
        print(resp)
    
    return jsonify({
        'success': True,
        'message': 'Debug data printed to console'
    })
# Add these debug routes to your Flask app (paste anywhere after your other @app.route definitions):

@app.route('/debug/trigger_deadline_check')
def trigger_deadline_check():
    """Manually trigger deadline checking for testing"""
    try:
        deadline_checker._check_expired_deadlines()
        return jsonify({'success': True, 'message': 'Deadline check triggered'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug/force_analytics/<int:user_id>')
def force_analytics(user_id):
    """Manually trigger analytics for testing"""
    try:
        waiting_key = f"waiting_for_responses_{user_id}"
        
        # Check session first
        from flask import session
        if waiting_key in session:
            waiting_state = session[waiting_key]
            result = handle_response_deadline_reached(user_id, waiting_state)
            return jsonify(result)
        else:
            return jsonify({'error': f'No waiting state found for user {user_id}', 'session_keys': list(session.keys())})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/check_timer/<int:user_id>')
def check_timer_status(user_id):
    """Check if timer expired and what should happen"""
    try:
        waiting_key = f"waiting_for_responses_{user_id}"
        from flask import session
        
        if waiting_key in session:
            waiting_state = session[waiting_key]
            deadline = datetime.strptime(waiting_state['deadline'], "%Y-%m-%d %H:%M:%S")
            current_time = datetime.now()
            
            return jsonify({
                'deadline': waiting_state['deadline'],
                'current_time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'time_remaining': str(deadline - current_time),
                'expired': current_time > deadline,
                'should_show_analytics': current_time > deadline,
                'event_id': waiting_state.get('event_id'),
                'session_key': waiting_key
            })
        else:
            return jsonify({
                'error': f'No waiting state found for user {user_id}',
                'all_session_keys': list(session.keys())
            })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/session_info')
def debug_session_info():
    """Show current session information"""
    from flask import session
    return jsonify({
        'session_keys': list(session.keys()),
        'session_data': {k: str(v) for k, v in session.items()}
    })

@app.route('/debug/mongodb_test')
def mongodb_test():
    try:
        # Test basic MongoDB connection and write capability
        test_doc = {
            'test_id': str(uuid.uuid4()),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': 'This is a test document'
        }
        
        # Try to insert a test document
        result = db.test_collection.insert_one(test_doc)
        
        # Try to read the document back
        retrieved = db.test_collection.find_one({'_id': result.inserted_id})
        
        # Get list of all collections
        collections = db.list_collection_names()
        
        return jsonify({
            'success': True,
            'connection': 'Working',
            'write_test': 'Successful',
            'inserted_id': str(result.inserted_id),
            'collections': collections,
            'document_retrieved': retrieved is not None
        })
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': trace
        })
@app.route('/get_available_file')
def get_available_file():
    """Find the best available dataset file"""
    try:
        # Try to find any clustered or generated dataset
        possible_files = [
            'clustered_generated_dataset.csv', 
            'clustered_dataset.csv',
            'generated_dataset.csv',
            'dataset.csv'
        ]
        
        # Also check for any uploaded datasets
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                         if f.endswith('.csv') and os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
        
        # Add any uploaded files to the beginning of the list
        possible_files = uploaded_files + possible_files
        
        for possible_file in possible_files:
            possible_path = os.path.join(app.config['UPLOAD_FOLDER'], possible_file)
            if os.path.exists(possible_path):
                return jsonify({
                    'success': True,
                    'filename': possible_file
                })
        
        # No files found
        return jsonify({
            'success': False,
            'message': 'No dataset files found'
        })
    
    except Exception as e:
        print(f"Error in get_available_file: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })


if __name__ == '__main__':
    # Initialize all database tables before starting the app
    ensure_tables_exist()
    deadline_checker.start() 
    app.run(debug=True,threaded=True)
