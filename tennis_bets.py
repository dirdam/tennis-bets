import streamlit as st
import BetChecker as bc

st.markdown('# Bet Checker')

if st.button('Get fixtures'):
    bet_checker = bc.BetChecker(headless=True)
    bet_checker.run()
    fixtures = bet_checker.show_fixtures()
    st.dataframe(fixtures, use_container_width=True)