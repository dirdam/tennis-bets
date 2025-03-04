import streamlit as st
import BetChecker as bc

st.set_page_config(
    page_title="Tennis Bets",
    page_icon="🎾"
    )

st.markdown('# Bet Checker')

if st.button('Get fixtures and ranks'):
    if 'bet_checker' not in st.session_state: # Create new BetChecker object if first time
        bet_checker = bc.BetChecker(headless=True)
    else: # Use existing BetChecker object if not first time
        bet_checker = st.session_state['bet_checker']
    bet_checker.run(next_hours=24)
    st.session_state['bet_checker'] = bet_checker

st.markdown('## Fixtures')
if 'bet_checker' in st.session_state:
    bet_checker = st.session_state['bet_checker']
    fixtures = bet_checker.fixtures
    cols = st.columns(2)
    with cols[0]:
        high_threshold = st.slider('High threshold (points):', min_value=1000, max_value=3000, value=2000, step=100)
    with cols[1]:
        low_threshold = st.slider('Low threshold (points)', min_value=0, max_value=high_threshold, value=1000, step=100)
    fixtures = bc.filter_fixtures(fixtures, high_threshold=high_threshold, low_threshold=low_threshold)
    fixtures = bc.beautify_fixtures(fixtures)
    st.dataframe(fixtures, use_container_width=True)