# Spotify Location Analyzer

This application analyzes the most popular music genres in different locations using the Spotify Web API. It provides a playlist of popular tracks from the specified location and identifies the most common genre.

## Features

- Search for music trends by location
- Display the most popular genre in the specified location
- Show a playlist of popular tracks from that location
- Preview tracks directly in the browser
- Modern, responsive UI with Spotify-inspired design

## Prerequisites

- Python 3.8 or higher
- Spotify Developer Account
- Spotify API credentials (Client ID and Client Secret)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd spotify_location_analyzer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Spotify API credentials:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

To get your Spotify API credentials:
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. Copy the Client ID and Client Secret
4. Add `http://localhost:5000` to your application's Redirect URIs

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a location (e.g., "Galway, Ireland") and click "Analyze"

## How It Works

The application:
1. Takes a location input from the user
2. Searches for tracks associated with that location
3. Analyzes the genres of the artists behind those tracks
4. Identifies the most common genre
5. Creates a playlist of the most popular tracks
6. Displays the results with preview functionality

## Note

The location-based search is based on Spotify's metadata and may not always perfectly match real-world location data. The results are based on tracks that are tagged with or associated with the specified location. 