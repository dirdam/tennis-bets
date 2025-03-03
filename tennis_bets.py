import streamlit as st
import BetChecker as bc

st.markdown('# Bet Checker')

next_hours = st.slider('Get fixtures within the next hours:', min_value=1, max_value=24, value=24, step=1)
if st.button('Get fixtures & Ranks'):
    bet_checker = bc.BetChecker(headless=True)
    bet_checker.run(next_hours=next_hours)
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
    # Show all rows with column 'Bettable' == True and top 5 rows with column 'Bettable' == False
    fix1 = fixtures[fixtures['Bettable'] == True]
    fix2 = fixtures[fixtures['Bettable'] == False].head(5)
    # Join both dataframes concatenated
    fixtures = pd.concat([fix1, fix2])
    st.dataframe(fixtures, use_container_width=True)