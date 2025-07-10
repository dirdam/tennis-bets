import os
import streamlit as st
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import json
import utils
import pandas as pd

# ---------- CONFIG ----------
PASSWORD = st.secrets["PASSWORD"]
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
DATASET = "dirdam/tennis-history"
TARGET_FILENAME1 = "players_history.json" 
TARGET_FILENAME2 = "matches.csv"
# ----------------------------

# Initialize Kaggle API
def download_dataset():
    api = KaggleApi()
    api.authenticate()

    msg = st.info("Downloading dataset from Kaggle...")
    api.dataset_download_files(DATASET, path=".", unzip=False)

    zip_path = f"{DATASET.split('/')[-1]}.zip"
    # Remove old TARGET_FILENAME1 if it exists
    target_path = os.path.join("data", TARGET_FILENAME1)
    if os.path.exists(target_path):
        os.remove(target_path)
    target_path2 = os.path.join("data", TARGET_FILENAME2)
    if os.path.exists(target_path2):
        os.remove(target_path2)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")

    # Remove the zip file after extraction
    if os.path.exists(zip_path):
        os.remove(zip_path)

    msg.empty()  # Remove the info message
    st.success("Download complete!")

# Password protection
def login():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        pwd = st.text_input("Enter password:", type="password")
        if pwd != PASSWORD:
            st.warning("Enter password to access the app.")
            st.stop()
        else:
            st.session_state['authenticated'] = True
    else:
        # Already authenticated, do nothing
        pass

if 'first_run' not in st.session_state:
    st.session_state['first_run'] = True

def main():
    global first_run

    st.title("🎾 Tennis strengths")
    
    login()

    if st.session_state['first_run']:
        download_dataset()
        st.session_state['first_run'] = False

    # Load data
    with open(f"data/{TARGET_FILENAME1}", 'r') as file:
        data = json.load(file)
    matches_df = pd.read_csv(f"data/{TARGET_FILENAME2}")
    st.markdown(f"(Last updated: {pd.to_datetime(str(matches_df['date'].max())).strftime('%d-%m-%Y')})")

    # Streamlit search and dropdowns for player selection
    st.markdown("### Select players")

    # Dropdowns for player selection
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", sorted(data) or ["(no match)"])
    with col2:
        player2 = st.selectbox("Player 2", sorted(data) or ["(no match)"])

    # Simulate match
    col1, col2 = st.columns(2)
    button_clicked = False
    past_matches_to_consider = 10 # Last X matches to consider
    with col1:
        if st.button("Simulate match of **3** sets", use_container_width=True):
            num_sets = 3
            button_clicked = True
    with col2:
        if st.button("Simulate match of **5** sets", use_container_width=True):
            num_sets = 5
            button_clicked = True
    if player1 == player2:
        st.warning("Please select two different players.")
    elif button_clicked: # Run simulations
        st.session_state['results'] = {}
        st.session_state['recent_data'] = {}
        progress_bar = st.progress(0, text="Simulating Monte-Carlo...")
        for i in range(1, past_matches_to_consider + 1):
            recent_data = {p: {'serve': data[p]['serve'][-i:], 'return': data[p]['return'][-i:]} for p in [player1, player2]}
            st.session_state['recent_data'][i] = utils.flatten_data(recent_data)
            results = utils.simulate_monte_carlo(player1, player2, st.session_state['recent_data'][i], num_sets=num_sets)
            st.session_state['results'][i] = results
            progress_bar.progress(i / past_matches_to_consider, text=f"Simulating Monte-Carlo... ({100 * i / past_matches_to_consider:.0f}%)")
        progress_bar.empty()

    if 'recent_data' in st.session_state and player1 in st.session_state['recent_data'][1] and player2 in st.session_state['recent_data'][1]:
        if 'matches_to_consider' not in st.session_state:
            st.session_state['matches_to_consider'] = 3
        st.markdown(f"Choose the number of last matches to consider:")
        matches_cols = st.columns(past_matches_to_consider)
        # Display a button to select how many matches to consider, one for each column
        for i in range(past_matches_to_consider):
            with matches_cols[i]:
                if st.button(f"{i + 1}", key=f"match_{i + 1}", use_container_width=True):
                    st.session_state['matches_to_consider'] = i + 1
 
        st.markdown(f"### Results for the last {st.session_state['matches_to_consider']} matches")
        # Horizontal bar showing win percentages
        utils.plot_horizontal_win_bar(player1, player2, st.session_state['results'][st.session_state['matches_to_consider']]['player1_wins_percent'], st.session_state['results'][st.session_state['matches_to_consider']]['player2_wins_percent'])

        # Display sets distribution
        utils.plot_sets_distribution(st.session_state['results'][st.session_state['matches_to_consider']]['sets_distribution'], player1, player2)

        # Display games count
        utils.plot_games_count(st.session_state['results'][st.session_state['matches_to_consider']]['games_count'])

        # Show strengths (optional)
        st.markdown(f"### Strengths")
        # Plot empirical comparison
        with st.expander(f"Graphical comparison for the last **{st.session_state['matches_to_consider']}** matches", expanded=False):
            utils.plot_empirical_comparison_side_by_side(st.session_state['recent_data'][st.session_state['matches_to_consider']], players_names=[player1, player2])


if __name__ == "__main__":
    main()
