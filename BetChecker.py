import streamlit as st
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd

def action_xpath(browser, xpath, action, others='', wait=20):
    others = '' if others == '' else "'''" + others + "'''"
    if wait > 0:
        WebDriverWait(browser, wait).until(EC.presence_of_element_located((By.XPATH, xpath)))
    eval(f"browser.find_element('xpath', '{xpath}').{action}({others})")

def remove_accents(input_str):
    import unicodedata
    # Normalize the string to decompose accentuated characters into base characters and accents
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    # Keep only base characters (e.g., remove combining accents)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

class BetChecker():
    def __init__(self, headless=True):
        self.headless = headless
        self.open_browser()

    def open_browser(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()) # Service(ChromeDriverManager().install())
        # service = Service(ChromeDriverManager().install())
        self.browser = webdriver.Chrome(service=service, options=options)
        self.browser.maximize_window()

    def close_browser(self):
        self.browser.quit()

    def get_fixtures_old(self):
        """OLD: removed because click is not properly handled in Streamlit"""
        """Gets all upcoming fixtures from TNNS"""
        from datetime import datetime
        progress_text = 'Getting fixtures...'
        my_bar = st.progress(0, text=progress_text) # Initialize progress bar
        print(progress_text)
        self.browser.get('https://tnnslive.com/')
        action_xpath(self.browser, '//*[@id="root"]/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/div[2]/div[2]/div/div[1]/div/div[1]/div/div/div/div[2]/div/div[1]/div[2]/div/div/div[4]/div/div', 'click')
        table = self.browser.find_element('xpath', '//*[@id="root"]/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/div[2]/div[2]/div/div[1]/div/div[1]/div/div/div/div[2]/div/div[2]')
        rows = table.find_elements('xpath', './/div[@tabindex="0"]')
        data = []
        last_tournament = ''
        last_category = ''
        today = datetime.today().date()
        for i, row in enumerate(rows):
            my_bar.progress((i+1)/len(rows), text=f'{progress_text} ({i+1}/{len(rows)})') # Update progress bar
            # Tournament info
            try:
                tournament_name = row.find_element('xpath', './/div[@class="css-901oao"]').text
                tournament_category = row.find_element('xpath', './/div[@class="css-901oao r-1jkjb r-1mnahxq"]').text
                if tournament_name != '':
                    last_tournament = tournament_name
                    last_category = tournament_category
            except:
                pass

            # Match details
            player_names = []
            try:
                players = row.find_elements('xpath', './/div[@dir="auto" and @class="css-901oao r-cqee49 r-vbi3md"]')
                player_names = [player.text for player in players]
                player1, player2 = player_names
                if '/' in player1:
                    continue
                match_time = row.find_element('xpath', './/div[@class="css-901oao r-cqee49 r-1smb3hh r-vbi3md r-1p6iasa r-nzoivv"]').text
                match_round = row.find_element('xpath', './/div[@class="css-901oao r-cqee49 r-1smb3hh r-1ws2f5x r-1p6iasa r-nzoivv"]').text
            except:
                pass

            if player_names != []:
                category_details = last_category.split(' · ')
                category = category_details[0]
                surface = category_details[-1]
                data.append({
                    'Tournament': last_tournament,
                    'Category': category,
                    'Surface': surface,
                    'Player1': player1,
                    'Player2': player2,
                    'Time': f'{today} {match_time}',
                    'Round': match_round
                })

        self.fixtures = pd.DataFrame(data)
        self.fixtures['Time'] = pd.to_datetime(self.fixtures['Time'], format='%Y-%m-%d %I:%M%p')

    def get_fixtures(self, next_hours=24):
        """Gets all upcoming fixtures from the backend of TNNS"""
        import requests, json
        from datetime import datetime, timezone, timedelta
        progress_text = 'Getting fixtures...'
        print(progress_text)
        st.write(progress_text)
        url = "https://gen2-matches-daily-web-ysvbugl7mq-uc.a.run.app/"
        # Fetch the JSON data
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Parse JSON
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            return
        
        # Create DataFrame
        tournaments = data['sids']
        matches = data['all_matches']
        for match in matches: # Force matches to have round info
            if 'd_st' not in match or 's' not in match['d_st']:
                match['d_st'] = {'s': ''} # Add empty round if not present
        matches = [match for match in matches if 'finishedAt' not in match]
        df = pd.DataFrame(matches)
        df['Tournament'] = df['sid'].apply(lambda x: tournaments[str(x)]['t'])
        df['Category+Surface'] = df['sid'].apply(lambda x: tournaments[str(x)]['su'])
        df['Category'] = df['Category+Surface'].apply(lambda x: x.split(' · ')[0])
        df['Surface'] = df['Category+Surface'].apply(lambda x: x.split(' · ')[-1])
        df['Player1'] = df['p'].apply(lambda x: x[0]['n'])
        df['Player2'] = df['p'].apply(lambda x: x[1]['n'])
        df['Time'] = df['start_time_timestamp'].apply(lambda x: datetime.fromtimestamp(int(x)/1000, tz=timezone.utc) + timedelta(hours=9)) # Japan time
        df['Time'] = df['Time'].dt.tz_localize(None)
        df['Round'] = df['d_st'].apply(lambda x: x['s'])
        df['Odd1'] = df['od'].apply(lambda x: x['o'][0]).astype(float).fillna(0)
        df['Odd2'] = df['od'].apply(lambda x: x['o'][1]).astype(float).fillna(0)
        df = df[['Tournament', 'Category', 'Surface', 'Player1', 'Player2', 'Odd1', 'Odd2', 'Time', 'Round']]
        # Restrict to times within 24 hours
        df = df[df['Time'] < datetime.now() + timedelta(hours=next_hours)]
        self.fixtures = df
    
    def get_ranks(self):
        """Adds the ATP/WTA rank and points of each player"""
        if 'ranks' in self.__dict__:
            print('Using already extracted ranks...')
            return
        players = []
        for type in ['atp', 'wta']:
            progress_text = f'Getting {type.upper()} rankings...'
            my_bar = st.progress(0, text=progress_text) # Initialize progress bar
            print(progress_text)
            self.browser.get(f'https://live-tennis.eu/en/{type}-live-ranking')
            WebDriverWait(self.browser, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="u868"]/tbody')))
            table = self.browser.find_element('xpath', '//*[@id="u868"]/tbody')
            rows = table.find_elements('xpath', './/tr')
            for i, row in enumerate(rows):
                my_bar.progress((i+1)/len(rows), text=f'{progress_text} ({i+1}/{len(rows)})') # Update progress bar
                try:
                    player = row.find_element('xpath', './/td[4]').text
                    player = remove_accents(player)
                    rank = row.find_element('xpath', './/td[1]').text
                    points = row.find_element('xpath', './/td[7]').text
                    players.append({
                        'Player': player,
                        'Rank': rank,
                        'Points': points
                    })
                except:
                    pass
        players = pd.DataFrame(players)
        players['Rank'] = players['Rank'].astype(int)
        players['Points'] = players['Points'].astype(int)
        self.ranks = players # Save ranks as attribute

    def add_ranks(self):
        """Adds the ATP/WTA rank and points of each player"""
        players = self.ranks
        self.fixtures['Rank1'] = self.fixtures['Player1'].apply(lambda x: players[players['Player'] == x]['Rank'].values[0] if x in players['Player'].values else 9999)
        self.fixtures['Rank2'] = self.fixtures['Player2'].apply(lambda x: players[players['Player'] == x]['Rank'].values[0] if x in players['Player'].values else 9999)
        self.fixtures['Points1'] = self.fixtures['Player1'].apply(lambda x: players[players['Player'] == x]['Points'].values[0] if x in players['Player'].values else 0)
        self.fixtures['Points2'] = self.fixtures['Player2'].apply(lambda x: players[players['Player'] == x]['Points'].values[0] if x in players['Player'].values else 0)

    def run(self, next_hours=24):
        self.open_browser()
        self.get_fixtures(next_hours=next_hours)
        self.get_ranks()
        self.add_ranks()
        self.close_browser()

def filter_fixtures(df, high_threshold=2000, low_threshold=1000):
    """Sorts the fixtures by highest points"""
    # Add 'bettable' column
    temp = df.copy()
    temp['Highest Points'] = temp[['Points1', 'Points2']].max(axis=1)
    temp['Lowest Points'] = temp[['Points1', 'Points2']].min(axis=1)
    temp['Bettable'] = temp.apply(lambda x: True if x['Highest Points'] >= high_threshold and x['Lowest Points'] < low_threshold else False, axis=1)
    # Sort by highest points and bettable
    temp = temp.sort_values(['Bettable', 'Highest Points'], ascending=[False, False])
    temp = temp.drop(columns=['Highest Points', 'Lowest Points']).reset_index(drop=True)
    return temp

def beautify_fixtures(df):
    temp = df.copy()
    temp['Player1'] = '(' + temp['Rank1'].astype(str) + ') ' + temp['Player1'] + ' [' + temp['Points1'].astype(str) + ']'
    temp['Player2'] = '(' + temp['Rank2'].astype(str) + ') ' + temp['Player2'] + ' [' + temp['Points2'].astype(str) + ']'
    temp = temp[['Bettable', 'Tournament', 'Player1', 'Player2', 'Odd1', 'Odd2', 'Time', 'Category', 'Round']]
    temp[['Odd1', 'Odd2']] = temp[['Odd1', 'Odd2']].round(2)
    # Show all rows with column 'Bettable' == True and top 5 rows with column 'Bettable' == False
    fix1 = temp[temp['Bettable'] == True]
    fix2 = temp[temp['Bettable'] == False].head(5)
    # Join both dataframes concatenated
    temp = pd.concat([fix1, fix2])
    return temp