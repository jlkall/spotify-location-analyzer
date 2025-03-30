import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import math
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

class GenreAnalyzer:
    def __init__(self):
        self.genre_weights = defaultdict(float)
        self.track_count = 0
        self.location_cache = {}
        
    def calculate_time_weight(self, release_date: str) -> float:
        """Calculate weight based on release date (more recent = higher weight)"""
        try:
            # Convert release date to datetime
            release_year = int(release_date.split('-')[0])
            current_year = datetime.now().year
            years_old = current_year - release_year
            
            # Exponential decay weight: newer songs have more weight
            # Weight ranges from 1.0 (current year) to 0.1 (50+ years old)
            time_weight = math.exp(-0.05 * years_old)
            return min(max(time_weight, 0.1), 1.0)
        except:
            return 0.5  # Default weight if date parsing fails
    
    def calculate_popularity_weight(self, popularity: int) -> float:
        """Calculate weight based on track popularity (more popular = higher weight)"""
        # Convert 0-100 popularity score to 0.1-1.0 weight
        return 0.1 + (popularity * 0.009)
    
    def get_location_variants(self, location: str) -> List[str]:
        """Generate location search variants"""
        if location in self.location_cache:
            return self.location_cache[location]
        
        # Clean the location string
        location = location.strip()
        
        # Split location into parts and clean each part
        parts = [part.strip() for part in location.split(',')]
        
        # Location-specific popular artists and music scenes
        location_artists = {
            'tokyo': ['YOASOBI', 'Official HIGE DANdism', 'King Gnu', 'Kenshi Yonezu', 'RADWIMPS'],
            'shibuya': ['Eve', 'ZUTOMAYO', 'Vaundy', 'Ado', 'millennium parade'],
            'seoul': ['BTS', 'BLACKPINK', 'TWICE', 'EXO', 'Red Velvet'],
            'london': ['Ed Sheeran', 'Adele', 'The Beatles', 'Queen', 'Pink Floyd'],
            'manchester': ['Oasis', 'The Smiths', 'Joy Division', 'Stone Roses', 'New Order'],
            'liverpool': ['The Beatles', 'Echo & the Bunnymen', 'Gerry and the Pacemakers'],
            'berlin': ['Rammstein', 'Paul Kalkbrenner', 'Fritz Kalkbrenner', 'Boris Brejcha'],
            'paris': ['Daft Punk', 'Justice', 'Air', 'Phoenix', 'David Guetta'],
            'new york': ['The Strokes', 'Nas', 'Jay-Z', 'The Velvet Underground', 'Blondie'],
            'los angeles': ['Red Hot Chili Peppers', 'The Doors', 'N.W.A', 'Dr. Dre', 'Snoop Dogg'],
            'seattle': ['Nirvana', 'Pearl Jam', 'Soundgarden', 'Alice In Chains', 'Foo Fighters'],
            'nashville': ['Johnny Cash', 'Taylor Swift', 'Keith Urban', 'Dolly Parton'],
            'memphis': ['Elvis Presley', 'B.B. King', 'Justin Timberlake', 'Three 6 Mafia'],
            'detroit': ['Eminem', 'The White Stripes', 'Bob Seger', 'Madonna', 'Kid Rock'],
            'chicago': ['Kanye West', 'Earth Wind & Fire', 'Chicago', 'Smashing Pumpkins'],
            'atlanta': ['OutKast', 'T.I.', 'Ludacris', 'Future', 'Migos'],
            'miami': ['Pitbull', 'Gloria Estefan', 'DJ Khaled', 'Rick Ross', 'Flo Rida'],
            'kingston': ['Bob Marley', 'Sean Paul', 'Shaggy', 'Beenie Man', 'Vybz Kartel'],
            'rio de janeiro': ['Joao Gilberto', 'Antonio Carlos Jobim', 'Chico Buarque'],
            'stockholm': ['ABBA', 'Avicii', 'Swedish House Mafia', 'Roxette', 'The Cardigans'],
            'dublin': ['U2', 'The Dubliners', 'Thin Lizzy', 'The Cranberries', 'Van Morrison'],
            'glasgow': ['Simple Minds', 'Franz Ferdinand', 'Belle & Sebastian', 'Travis'],
            'amsterdam': ['Tiesto', 'Martin Garrix', 'Armin van Buuren', 'Andre Hazes'],
            'vienna': ['Mozart', 'Beethoven', 'Johann Strauss II', 'Falco', 'Kruder & Dorfmeister'],
            'sydney': ['AC/DC', 'INXS', 'Nick Cave', 'Empire of the Sun', 'Tame Impala']
        }
        
        variants = []
        # Add original location
        variants.append(location)
        
        # Get the main part (usually city/area name)
        main_part = parts[0]
        variants.append(main_part)
        
        # Add location-specific artists if available
        location_lower = main_part.lower()
        if location_lower in location_artists:
            for artist in location_artists[location_lower]:
                variants.extend([
                    artist,
                    f"{artist} {main_part}",
                    f"{main_part} {artist}"
                ])
        
        # Add English translation for common international cities
        common_translations = {
            'tokyo': ['東京', 'とうきょう', 'トウキョウ', 'J-POP', 'ジェイポップ'],
            'shibuya': ['渋谷', 'シブヤ', '渋谷系', 'シブヤ系'],
            'kyoto': ['京都', 'きょうと', 'キョウト'],
            'osaka': ['大阪', 'おおさか', 'オオサカ'],
            'beijing': ['北京', '베이징', 'C-POP'],
            'seoul': ['서울', '首尔', 'K-POP', '케이팝'],
            'paris': ['パリ', 'パリス', 'chanson'],
            'moscow': ['москва', 'Москва']
        }
        
        # Check if we have translations for this location
        if location_lower in common_translations:
            variants.extend(common_translations[location_lower])
        
        # Add variants without special characters
        cleaned_main = ''.join(c for c in main_part if c.isalnum() or c.isspace()).strip()
        if cleaned_main != main_part:
            variants.append(cleaned_main)
        
        # Add music-related variants
        music_variants = [
            f"music {main_part}",
            f"{main_part} music",
            f"{main_part} artists",
            f"songs from {main_part}",
            f"{main_part} scene",
            f"popular {main_part}",
            f"{main_part} hits",
            f"{main_part} songs",
            f"best {main_part}",
            f"{main_part} classics",
            f"{main_part} anthems"
        ]
        variants.extend(music_variants)
        
        # For international cities, add translated music-related terms
        if location_lower in common_translations:
            for translation in common_translations[location_lower]:
                variants.extend([
                    f"{translation} music",
                    f"music {translation}",
                    translation
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique_variants.append(v)
        
        self.location_cache[location] = unique_variants
        return unique_variants
    
    def get_tracks_for_location(self, location: str) -> List[Dict]:
        """Get tracks from multiple search variants"""
        all_tracks = []
        seen_track_ids = set()
        
        # Try each search variant
        for variant in self.get_location_variants(location):
            try:
                # Try different search combinations
                search_queries = [
                    variant,  # Direct search
                    f"track:{variant}",  # Track-specific search
                    f"artist:{variant}",  # Artist-specific search
                    f"album:{variant}"  # Album-specific search
                ]
                
                for query in search_queries:
                    try:
                        # Try multiple search types for better results
                        for search_type in ['track', 'artist']:
                            results = sp.search(q=query, type=search_type, limit=20)
                            
                            if search_type == 'track' and results and results['tracks']['items']:
                                for track in results['tracks']['items']:
                                    if track['id'] not in seen_track_ids:
                                        seen_track_ids.add(track['id'])
                                        all_tracks.append(track)
                            
                            elif search_type == 'artist' and results and results['artists']['items']:
                                # For artist searches, get their top tracks
                                for artist in results['artists']['items'][:3]:  # Limit to top 3 artists
                                    try:
                                        top_tracks = sp.artist_top_tracks(artist['id'])
                                        for track in top_tracks['tracks']:
                                            if track['id'] not in seen_track_ids:
                                                seen_track_ids.add(track['id'])
                                                all_tracks.append(track)
                                    except Exception as e:
                                        logger.warning(f"Error getting top tracks for artist {artist['name']}: {str(e)}")
                                        continue
                    
                    except Exception as e:
                        logger.warning(f"Error searching with query '{query}': {str(e)}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error searching for variant '{variant}': {str(e)}")
                continue
            
            # If we have enough tracks, stop searching
            if len(all_tracks) >= 20:
                break
        
        # If we still don't have enough tracks, try a broader search
        if len(all_tracks) < 5:
            try:
                # Try searching for popular tracks in the general area
                for query in [f"popular {location}", f"top {location}", location]:
                    results = sp.search(q=query, type='track', limit=20)
                    if results and results['tracks']['items']:
                        for track in results['tracks']['items']:
                            if track['id'] not in seen_track_ids:
                                seen_track_ids.add(track['id'])
                                all_tracks.append(track)
                            if len(all_tracks) >= 20:
                                break
                        if len(all_tracks) >= 20:
                            break
            except Exception as e:
                logger.error(f"Error performing fallback search: {str(e)}")
        
        return all_tracks[:20]  # Return up to 20 tracks
    
    def analyze_genres(self, location: str) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Analyze genres for a location with weighted scoring
        Returns: (genre_scores, playlist_tracks)
        """
        # Reset weights for new analysis
        self.genre_weights.clear()
        self.track_count = 0
        
        # Get tracks for location
        tracks = self.get_tracks_for_location(location)
        if not tracks:
            return {}, []
        
        # Collect all artist IDs (including featured artists)
        artist_ids = set()
        track_artists = defaultdict(list)
        
        for track in tracks:
            for artist in track['artists']:
                artist_ids.add(artist['id'])
                track_artists[track['id']].append(artist['id'])
        
        # Get artist details in batches of 50 (API limit)
        artist_details = {}
        for i in range(0, len(artist_ids), 50):
            batch_ids = list(artist_ids)[i:i+50]
            try:
                artists_batch = sp.artists(batch_ids)
                for artist in artists_batch['artists']:
                    if artist:  # Check if artist data exists
                        artist_details[artist['id']] = artist
            except Exception as e:
                logger.error(f"Error fetching artist details: {str(e)}")
        
        # Process each track and its artists
        playlist_tracks = []
        for track in tracks:
            try:
                # Calculate track-specific weights
                time_weight = self.calculate_time_weight(track['album']['release_date'])
                popularity_weight = self.calculate_popularity_weight(track['popularity'])
                
                # Combined weight for this track
                track_weight = time_weight * popularity_weight
                
                # Process all artists for the track
                track_genre_weight = track_weight / len(track_artists[track['id']])
                
                for artist_id in track_artists[track['id']]:
                    if artist_id in artist_details:
                        artist = artist_details[artist_id]
                        # Add weighted score to each genre
                        for genre in artist['genres']:
                            self.genre_weights[genre] += track_genre_weight
                
                # Add to playlist if it has a preview URL
                if track['preview_url'] and len(playlist_tracks) < 20:
                    playlist_tracks.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'preview_url': track['preview_url'],
                        'popularity': track['popularity'],
                        'release_date': track['album']['release_date']
                    })
                
                self.track_count += 1
                
            except Exception as e:
                logger.error(f"Error processing track {track.get('name', 'Unknown')}: {str(e)}")
        
        # Normalize genre weights
        if self.track_count > 0:
            genre_scores = {
                genre: weight / self.track_count 
                for genre, weight in self.genre_weights.items()
            }
        else:
            genre_scores = {}
        
        # Sort playlist by popularity
        playlist_tracks.sort(key=lambda x: x['popularity'], reverse=True)
        
        return genre_scores, playlist_tracks

    def get_genre_categories(self, genre_scores: Dict[str, float], top_n: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Categorize genres into main music styles
        Returns top N genres in each category
        """
        categories = {
            'electronic': ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep'],
            'hip_hop': ['hip hop', 'rap', 'trap', 'grime'],
            'rock': ['rock', 'metal', 'punk', 'indie', 'alternative'],
            'pop': ['pop', 'dance pop', 'indie pop', 'synth pop'],
            'folk': ['folk', 'acoustic', 'singer-songwriter'],
            'jazz': ['jazz', 'swing', 'bebop', 'fusion'],
            'classical': ['classical', 'orchestra', 'chamber', 'opera'],
            'world': ['world', 'latin', 'reggae', 'afrobeat', 'celtic']
        }
        
        categorized = defaultdict(list)
        
        for genre, score in genre_scores.items():
            for category, keywords in categories.items():
                if any(keyword in genre.lower() for keyword in keywords):
                    categorized[category].append((genre, score))
                    break
            else:
                categorized['other'].append((genre, score))
        
        # Sort each category by score and get top N
        return {
            category: sorted(genres, key=lambda x: x[1], reverse=True)[:top_n]
            for category, genres in categorized.items()
            if genres  # Only include non-empty categories
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_location():
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
        location = data.get('location')
    else:
        location = request.form.get('location')

    if not location:
        return jsonify({'error': 'Please provide a location'}), 400
    
    try:
        analyzer = GenreAnalyzer()
        genre_scores, playlist_tracks = analyzer.analyze_genres(location)
        
        if not genre_scores:
            return jsonify({'error': 'No data found for this location'}), 404
        
        # Get top genres by category
        categorized_genres = analyzer.get_genre_categories(genre_scores)
        
        # Get overall top genres
        top_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            'location': location,
            'top_genre': top_genres[0][0] if top_genres else None,
            'top_genres': [{'name': genre, 'score': float(score)} for genre, score in top_genres],
            'genre_categories': {
                category: [{'name': genre, 'score': float(score)} for genre, score in genres]
                for category, genres in categorized_genres.items()
            },
            'playlist': playlist_tracks
        })
        
    except Exception as e:
        logger.error(f"Error analyzing location {location}: {str(e)}")
        return jsonify({'error': 'Could not fetch music data for this location'}), 500

if __name__ == '__main__':
    app.run(debug=True) 