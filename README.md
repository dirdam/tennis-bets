# Tennis bets

https://tennis-bets.streamlit.app/

Steps the program does:
1. Get all upcoming matching within the next [XX] hours (defaults to 24 hours).
2. Get the ATP and WTA rankings.
3. Marks as `Bettable` those matches where one player has more than `High threshold` points and the other less than `Low threshold` points.