from collections import Counter
import plotly.graph_objs as go
import streamlit as st
import random
import numpy as np
import pandas as pd

player1_color = "skyblue"  # Light blue
# Light orange
player2_color = "lightsalmon"  # Light orange

def flatten_data(data):
    flattened = {}
    for player, stats in data.items():
        flattened[player] = {
            'serve': [item if abs(item) <= 4 else (4 if item > 4 else -4) for sublist in data[player]['serve'] for item in sublist],
            'return': [item if abs(item) <= 4 else (4 if item > 4 else -4) for sublist in data[player]['return'] for item in sublist]
        }
    return flattened

def get_empirical_probs(data, support=None):
    '''Calculates empirical probabilities for given support based on the data.'''
    counts = Counter(data)
    total = sum(counts.values())
    if support is None:
        support = sorted(counts.keys())
    probs = [counts.get(x, 0) / total for x in support]
    return support, probs

def plot_empirical_comparison_side_by_side(data, players_names, key=None):
    '''Plots side-by-side comparisons of 'serve' and 'return' empirical distributions between two players using Plotly and Streamlit.'''
    categories = ['serve', 'return']
    for idx, category in enumerate(categories):
        data1 = data[players_names[0]][category]
        data2 = data[players_names[1]][category]

        # Get the full integer range between min and max of both datasets, then remove 1 and -1
        min_val = int(min(min(data1), min(data2)))
        max_val = int(max(max(data1), max(data2)))
        full_support = [x for x in range(min_val, max_val + 1) if x not in (1, -1)]

        support1, probs1 = get_empirical_probs(data1, full_support)
        support2, probs2 = get_empirical_probs(data2, full_support)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=support1,
            y=probs1,
            name=players_names[0],
            opacity=0.7,
            marker=dict(color=player1_color)
        ))
        fig.add_trace(go.Bar(
            x=support2,
            y=probs2,
            name=players_names[1],
            opacity=0.7,
            marker=dict(color=player2_color)
        ))

        fig.update_layout(
            barmode='group',
            title=f"{category.capitalize()} distribution comparison",
            xaxis_title="Value",
            yaxis_title="Empirical probability",
            legend_title="Player",
            xaxis=dict(
            tickmode='linear',
            tick0=min_val,
            dtick=1
            )
        )
        chart_key = f"{key}_cat_{idx}" if key is not None else None
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

def generate_from_empirical(data):
    '''Generates a random sample from empirical data using weighted probabilities.'''
    counts = Counter(data)
    support = list(counts.keys())
    total = sum(counts.values())
    probabilities = [counts[x] / total for x in support]
    return random.choices(support, weights=probabilities, k=1)[0]

def simulate_game(server, receiver, data):
    '''Simulate a game between server and receiver based on their predicted stats. Returns the winner of the game.'''
    # Get value from distributions
    points_diff_server = generate_from_empirical(data[server]['serve'])
    points_diff_receiver = generate_from_empirical(data[receiver]['return'])
    # Check winner
    return server if points_diff_server > points_diff_receiver else (receiver if points_diff_receiver > points_diff_server else random.choice([server, receiver])) # Random choice if tie

def simulate_set(player1, player2, data):
    '''Simulate a set between player1 and player2. Returns the number of games won by each player.'''
    player1_games = 0
    player2_games = 0
    server = player1
    while max(player1_games, player2_games) < 6 or (max(player1_games, player2_games) - min(player1_games, player2_games) < 2):
        receiver = player2 if server == player1 else player1
        # Simulate a game
        winner = simulate_game(server, receiver, data)
        if winner == player1:
            player1_games += 1
        else:
            player2_games += 1
        # Switch server
        server = receiver
        if max(player1_games, player2_games) == 7: # If one player reaches 7 games, they win the set
            break
    return {player1: player1_games, player2: player2_games}

def simulate_match(player1, player2, data, num_sets=3, verbose=False):
    '''Simulate a match between player1 and player2. Returns the number of sets won by each player.'''
    player1_sets = 0
    player2_sets = 0
    server = player1
    total_games = 0
    while max(player1_sets, player2_sets) < (num_sets // 2 + 1):
        receiver = player2 if server == player1 else player1
        # Simulate a set
        set_result = simulate_set(server, receiver, data)
        if verbose:
            print(f"Set result: {set_result}")
        if set_result[player1] > set_result[player2]:
            player1_sets += 1
        else:
            player2_sets += 1
        # Track total games played
        total_games += sum(set_result.values())
        # Switch server
        server = receiver
    return {player1: player1_sets, player2: player2_sets, 'total_games': total_games}

def simulate_monte_carlo(player1, player2, recent_data, num_sets, num_matches=10000, st_progress=True):
    """Simulate a large number of matches and return win counts, percentages, and set distributions."""
    player1_wins = 0
    player2_wins = 0
    server = player1 if np.random.rand() < 0.5 else player2  # Randomly choose server for the first match
    sets_distribution = {}  # Example: {'2-0': 5000, '2-1': 3000, '1-2': 2000} to track set results
    games_count = {}  # Track games count for each match
    progress_bar = st.progress(0, text="Simulating matches...") if st_progress else None
    for i in range(num_matches):
        receiver = player2 if server == player1 else player1
        # Simulate a match
        match_result = simulate_match(server, receiver, recent_data, num_sets=num_sets)
        if match_result[player1] > match_result[player2]:
            player1_wins += 1
        else:
            player2_wins += 1
        # Track sets distribution
        sets_result = f"{match_result[player1]}-{match_result[player2]}"
        if sets_result not in sets_distribution:
            sets_distribution[sets_result] = 1
        else:
            sets_distribution[sets_result] += 1
        # Track games count
        total_games = match_result['total_games']
        if total_games not in games_count:
            games_count[total_games] = 1
        else:
            games_count[total_games] += 1
        # Switch server
        server = receiver  # Alternate server for the next match
        if st_progress and (i % max(1, num_matches // 100) == 0 or i == num_matches - 1):
            progress_bar.progress((i + 1) / num_matches, text=f"Simulating matches... ({100*(i + 1)/num_matches:.0f}%)")
    if st_progress:
        progress_bar.empty()
    player1_wins_percent = (player1_wins / num_matches) * 100
    player2_wins_percent = (player2_wins / num_matches) * 100
    return {
        'player1_wins': player1_wins,
        'player2_wins': player2_wins,
        'player1_wins_percent': player1_wins_percent,
        'player2_wins_percent': player2_wins_percent,
        'sets_distribution': sets_distribution,
        'games_count': games_count
    }

def plot_horizontal_win_bar(player1, player2, player1_wins_percent, player2_wins_percent, key=None):
    """Plots a single horizontal bar split by win percentage for two players using Plotly and Streamlit."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[player1_wins_percent],
        y=[''],
        orientation='h',
        name=player1,
        marker=dict(color=player1_color),
        text=f"{player1} ({player1_wins_percent:.2f}%)",
        textposition='inside',
        insidetextanchor='middle',
        showlegend=False
    ))
    fig.add_trace(go.Bar(
        x=[player2_wins_percent],
        y=[''],
        orientation='h',
        name=player2,
        marker=dict(color=player2_color),
        text=f"{player2} ({player2_wins_percent:.2f}%)",
        textposition='inside',
        insidetextanchor='middle',
        showlegend=False
    ))
    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 100], title='Win percentage'),
        yaxis=dict(showticklabels=False),
        height=100,
        margin=dict(l=30, r=30, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_sets_distribution(sets_distribution, key=None):
    def custom_sort_key(item):
        '''Custom sort: winner sets descending, loser sets ascending (e.g., 3-0, 3-1, 3-2, 2-3, 1-3, 0-3)'''
        winner, loser = map(int, item[0].split('-'))
        return (-winner, loser)
    sorted_distribution = sorted(sets_distribution.items(), key=custom_sort_key)
    sets_labels = [sets[0] for sets in sorted_distribution]
    sets_values = [sets[1] for sets in sorted_distribution]
    total_matches = sum(sets_values)
    percentages = [value / total_matches * 100 for value in sets_values]

    # Assign colors: player1_color for '3-X', player2_color for 'X-3', default for others
    bar_colors = []
    for label in sets_labels:
        player1_result, player2_result = map(int, label.split('-'))
        if max(player1_result, player2_result) == player1_result:
            bar_colors.append(player1_color)
        else:
            bar_colors.append(player2_color)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sets_labels,
        y=sets_values,
        text=[f"{pct:.2f}%" for pct in percentages],
        textposition='outside',
        marker_color=bar_colors
    ))
    fig.update_layout(
        title='Distribution of sets',
        xaxis_title='Sets result',
        yaxis_title='Number of matches',
        xaxis_tickangle=-45,
        bargap=0.2,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_games_count(games_count, key=None):
    """Plots the distribution of total games played in simulated matches using Plotly and Streamlit, with vertical lines for mean and median."""
    x_vals = list(games_count.keys())
    y_vals = list(games_count.values())

    # Compute mean and median
    all_games = []
    for games, count in games_count.items():
        all_games.extend([games] * count)
    mean_games = np.mean(all_games)
    median_games = np.median(all_games)

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_vals,
                y=y_vals,
                marker_color='skyblue',
                name='Games count'
            ),
            go.Scatter(
                x=[mean_games, mean_games],
                y=[0, max(y_vals)],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f"Mean: {mean_games:.2f}",
                showlegend=True
            ),
            go.Scatter(
                x=[median_games, median_games],
                y=[0, max(y_vals)],
                mode='lines',
                line=dict(color='green', dash='dot'),
                name=f"Median: {median_games:.2f}",
                showlegend=True
            )
        ]
    )
    fig.update_layout(
        title='Distribution of total games',
        xaxis_title='Total games',
        yaxis_title='Number of matches',
        bargap=0.2,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_xaxes(tickangle=45, showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    st.plotly_chart(fig, use_container_width=True, key=key)

def calculate_prediction_differences(results):
    """Calculates the differences in win percentages between two players across multiple simulations."""
    diff = []
    for i in range(1, len(results) + 1):
        diff.append(results[i]['player1_wins_percent'] - results[i]['player2_wins_percent'])
    return diff

def plot_prediction_differences(results, player1, player2):
    """Plots the differences in win percentages between two players across multiple simulations as a vertical plot, with x-axis reversed.
    The line is thicker near the top and slimmer near the bottom."""
    prediction_differences = calculate_prediction_differences(results)
    y_vals = list(range(1, len(prediction_differences) + 1))
    # Create a list of line widths: thickest at the top, slimmest at the bottom
    max_width = 8
    min_width = 2
    n = len(y_vals)
    # Linear interpolation from max_width to min_width
    widths = [max_width - (max_width - min_width) * (i / (n - 1)) if n > 1 else max_width for i in range(n)]

    fig = go.Figure()
    # Plot as segments to vary line width and color by top value
    for i in range(1, n):
        top_value = prediction_differences[i-1]
        color = player1_color if top_value >= 0 else player2_color
        fig.add_trace(go.Scatter(
            x=prediction_differences[i-1:i+1],
            y=y_vals[i-1:i+1],
            mode='lines',
            line=dict(color=color, width=widths[i-1]),
            showlegend=False
        ))
    # Add markers matching the size and color of the segments
    marker_sizes = widths
    marker_colors = [player1_color if diff >= 0 else player2_color for diff in prediction_differences]
    fig.add_trace(go.Scatter(
        x=prediction_differences,
        y=y_vals,
        mode='markers',
        marker=dict(size=marker_sizes, color=marker_colors),
        showlegend=False
    ))
    fig.update_layout(
        title=f"Prediction differences",
        yaxis_title="Number of matches considered",
        xaxis_title="Win percentage difference",
        yaxis=dict(dtick=1, autorange='reversed'),
        xaxis=dict(
            range=[100, -100],
            tickvals=[100, 75, 50, 25, 0, -25, -50, -75, -100],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgray',
            showgrid=True,
            gridcolor='lightgray',
            griddash='dash'  # Make vertical grid lines dashed
        ),
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )
    fig.add_annotation(
        x=100, y=1, text=f"<b>{player1}</b>", showarrow=False, xanchor='left', yanchor='bottom', font=dict(size=14, color=player1_color)
    )
    fig.add_annotation(
        x=-100, y=1, text=f"<b>{player2}</b>", showarrow=False, xanchor='right', yanchor='bottom', font=dict(size=14, color=player2_color)
    )
    st.plotly_chart(fig, use_container_width=True)

def get_last_matches(matches_df, player1, player2):
    """Returns the last match records of two players from the matches DataFrame."""
    def get_last_record(df, player):
        records = df[(df['winner'] == player) | (df['loser'] == player)]
        if records.empty:
            return pd.DataFrame()
        last_record = records.iloc[:1].copy()
        last_record['player'] = player
        last_record['last_rival'] = last_record['loser'] if last_record['winner'].iloc[0] == player else last_record['winner']
        last_record['result'] = 'Won' if last_record['winner'].iloc[0] == player else 'Lost'
        return last_record

    player1_last = get_last_record(matches_df, player1)
    player2_last = get_last_record(matches_df, player2)
    both_players = pd.concat([player1_last, player2_last], ignore_index=True).sort_values(by='date', ascending=False)
    both_players['date'] = both_players['date'].apply(lambda d: pd.to_datetime(str(d)).strftime('%d-%m-%Y'))
    return both_players[['player', 'tournament', 'stage', 'last_rival', 'result', 'date']]
