from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import pandas as pd
import streamlit as st

def action_xpath(browser, xpath, action, others='', wait=60):
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
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()) # Service(ChromeDriverManager().install())
        self.browser = webdriver.Chrome(service=service, options=options)
        self.browser.maximize_window()

    def close_browser(self):
        self.browser.quit()

    def get_fixtures(self):
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
        for i, row in tqdm(enumerate(rows)):
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

    def show_fixtures(self):
        """Returns all upcoming fixtures"""
        return self.fixtures
    
    def add_rank(self):
        """Adds the ATP/WTA rank and points of each player"""
        players = []
        for type in ['atp', 'wta']:
            progress_text = f'Getting {type.upper()} rankings...'
            my_bar = st.progress(0, text=progress_text) # Initialize progress bar
            print(progress_text)
            self.browser.get(f'https://live-tennis.eu/en/{type}-live-ranking')
            WebDriverWait(self.browser, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="u868"]/tbody')))
            table = self.browser.find_element('xpath', '//*[@id="u868"]/tbody')
            rows = table.find_elements('xpath', './/tr')
            for i, row in tqdm(enumerate(rows)):
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
        self.fixtures['Rank1'] = self.fixtures['Player1'].apply(lambda x: players[players['Player'] == x]['Rank'].values[0] if x in players['Player'].values else 9999)
        self.fixtures['Rank2'] = self.fixtures['Player2'].apply(lambda x: players[players['Player'] == x]['Rank'].values[0] if x in players['Player'].values else 9999)
        self.fixtures['Points1'] = self.fixtures['Player1'].apply(lambda x: players[players['Player'] == x]['Points'].values[0] if x in players['Player'].values else 0)
        self.fixtures['Points2'] = self.fixtures['Player2'].apply(lambda x: players[players['Player'] == x]['Points'].values[0] if x in players['Player'].values else 0)

    def sort_fixtures(self, high_threshold=2000, low_threshold=1000):
        """Sorts the fixtures by highest points"""
        # Add 'bettable' column
        self.fixtures['Highest Points'] = self.fixtures[['Points1', 'Points2']].max(axis=1)
        self.fixtures['Lowest Points'] = self.fixtures[['Points1', 'Points2']].min(axis=1)
        self.fixtures['Bettable'] = self.fixtures.apply(lambda x: True if x['Highest Points'] >= high_threshold and x['Lowest Points'] < low_threshold else False, axis=1)
        # Sort by highest points and bettable
        self.fixtures = self.fixtures.sort_values(['Bettable', 'Highest Points'], ascending=[False, False])
        self.fixtures = self.fixtures.drop(columns=['Highest Points', 'Lowest Points']).reset_index(drop=True)
    
    def beautify_fixtures(self):
        self.fixtures['Player1'] = '(' + self.fixtures['Rank1'].astype(str) + ') ' + self.fixtures['Player1']
        self.fixtures['Player2'] = '(' + self.fixtures['Rank2'].astype(str) + ') ' + self.fixtures['Player2']
        self.fixtures = self.fixtures[['Bettable', 'Player1', 'Player2', 'Time', 'Tournament', 'Category', 'Round']]
        def color_rows(row):
            color = 'background-color: lightgreen' if row['Bettable'] else 'background-color: lavenderblush'
            return [color] * len(row)
        self.fixtures = self.fixtures.style.apply(color_rows, axis=1)

    def run(self):
        self.open_browser()
        self.get_fixtures()
        self.add_rank()
        self.close_browser()
        self.sort_fixtures()
        self.beautify_fixtures()