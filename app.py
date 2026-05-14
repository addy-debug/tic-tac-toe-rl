# app.py
# Q-Learning Tic Tac Toe AI with Streamlit
# Q-Learning + NumPy + Streamlit with Enhanced UI

import streamlit as st
import numpy as np
import random
import time

st.set_page_config(
    page_title="Tic Tac Toe AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS STYLING
# =====================================================

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
    }
    
    .stContainer {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        text-shadow: 0 4px 20px rgba(56, 189, 248, 0.3);
        letter-spacing: -1px;
    }
    
    h2 {
        color: #38bdf8 !important;
        font-size: 1.8rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-size: 1.3rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Card Styling */
    .card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    /* Game Board */
    .game-board {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
        margin: 1.5rem 0;
    }
    
    /* Cell Styling */
    .cell-button {
        aspect-ratio: 1;
        border: none;
        border-radius: 15px;
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        color: #38bdf8;
        font-size: 3rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(56, 189, 248, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .cell-button:hover:not(:disabled) {
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
        border: 2px solid rgba(56, 189, 248, 0.5);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4);
        transform: translateY(-4px) scale(1.02);
    }
    
    .cell-button:active:not(:disabled) {
        transform: translateY(-2px) scale(0.98);
    }
    
    .cell-button:disabled {
        opacity: 1;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Alert/Success/Error Styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.1) 0%, rgba(34, 197, 94, 0.1) 100%) !important;
        border-left: 4px solid #4ade80 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        color: #86efac !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        color: #fca5a5 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%) !important;
        border-left: 4px solid #60a5fa !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        color: #93c5fd !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    /* Metrics Styling */
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 2px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stMetric > label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stMetric > div > div > div > div:first-child > div:first-child {
        color: #38bdf8 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Column Separator */
    .column-divider {
        border-left: 2px solid rgba(56, 189, 248, 0.2);
        margin: 0 2rem;
    }
    
    /* Progress Bar */
    .progress-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%);
        height: 8px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }
    
    /* Text */
    p {
        color: #cbd5e1 !important;
        line-height: 1.6 !important;
    }
    
    /* Links */
    a {
        color: #38bdf8 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #0ea5e9 !important;
        text-decoration: underline !important;
    }
    
    /* Markdown */
    .markdown-text-container {
        color: #cbd5e1 !important;
    }
    
    ul {
        color: #cbd5e1 !important;
    }
    
    li {
        margin: 0.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# Q LEARNING AGENT
# =====================================================

class QLearningAgent:

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2):

        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        return self.q_table[state]

    def choose_action(self, state, available_moves):

        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)

        # Exploitation
        q_values = self.get_q_values(state)

        best_move = available_moves[0]
        best_score = -999

        for move in available_moves:

            if q_values[move] > best_score:
                best_score = q_values[move]
                best_move = move

        return best_move

    def update_q_value(self, state, action, reward, next_state):

        current_q = self.get_q_values(state)[action]

        future_q = np.max(self.get_q_values(next_state))

        new_q = current_q + self.alpha * (
            reward + self.gamma * future_q - current_q
        )

        self.q_table[state][action] = new_q


# =====================================================
# GAME LOGIC
# =====================================================

class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        self.winner = None

    def available_moves(self):

        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, position, player):

        if self.board[position] != " ":
            return False

        self.board[position] = player

        if self.check_winner(player):
            self.winner = player

        return True

    def check_winner(self, player):

        winning_combinations = [

            [0,1,2],
            [3,4,5],
            [6,7,8],

            [0,3,6],
            [1,4,7],
            [2,5,8],

            [0,4,8],
            [2,4,6]
        ]

        for combo in winning_combinations:

            if all(self.board[i] == player for i in combo):
                return True

        return False

    def is_draw(self):

        return " " not in self.board and self.winner is None

    def get_state(self):

        return "".join(self.board)


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

if "agent_x" not in st.session_state:
    st.session_state.agent_x = QLearningAgent()
    st.session_state.agent_o = QLearningAgent()
    st.session_state.human_game = TicTacToe()
    st.session_state.stats = {
        "episodes": 0,
        "x_wins": 0,
        "o_wins": 0,
        "draws": 0
    }
    st.session_state.training_count = 0
    st.session_state.total_games_played = 0


# =====================================================
# AI TRAINING (Run on each app load)
# =====================================================

def train_ai(iterations=100):
    for _ in range(iterations):
        training_game = TicTacToe()
        current_player = "X"

        while True:
            state = training_game.get_state()
            available = training_game.available_moves()

            if current_player == "X":
                action = st.session_state.agent_x.choose_action(state, available)
            else:
                action = st.session_state.agent_o.choose_action(state, available)

            training_game.make_move(action, current_player)
            next_state = training_game.get_state()

            reward = 0

            # WIN
            if training_game.winner == current_player:
                reward = 1
                if current_player == "X":
                    st.session_state.stats["x_wins"] += 1
                else:
                    st.session_state.stats["o_wins"] += 1

            # DRAW
            elif training_game.is_draw():
                reward = 0.5
                st.session_state.stats["draws"] += 1

            # UPDATE Q TABLE
            if current_player == "X":
                st.session_state.agent_x.update_q_value(state, action, reward, next_state)
            else:
                st.session_state.agent_o.update_q_value(state, action, reward, next_state)

            # END GAME
            if training_game.winner or training_game.is_draw():
                st.session_state.stats["episodes"] += 1
                break

            # SWITCH PLAYER
            current_player = "O" if current_player == "X" else "X"


# Train AI in background
if st.session_state.training_count < 1000:
    train_ai(10)
    st.session_state.training_count += 10


# =====================================================
# UI
# =====================================================

st.markdown("""<h1>🎮 Tic Tac Toe AI 🤖</h1>""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""<div class="card">""", unsafe_allow_html=True)
    st.markdown("""<h2>🎯 Play Against AI</h2>""", unsafe_allow_html=True)
    
    # Game board in a container
    st.markdown("""<div class="game-board" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; padding: 2rem; background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%); border-radius: 20px;">""", unsafe_allow_html=True)
    
    board_cols = st.columns(3)
    for i in range(3):
        board_rows = st.columns(3)
        for j in range(3):
            idx = i * 3 + j
            with board_rows[j]:
                cell_value = st.session_state.human_game.board[idx]
                
                # Display X and O with colors
                if cell_value == "X":
                    display_text = "❌"
                    button_disabled = True
                elif cell_value == "O":
                    display_text = "⭕"
                    button_disabled = True
                else:
                    display_text = "•"
                    button_disabled = st.session_state.human_game.winner is not None
                
                if st.button(
                    display_text,
                    key=f"cell_{idx}",
                    use_container_width=True,
                    disabled=button_disabled,
                    help=f"Position {idx + 1}"
                ):
                    # Human move
                    st.session_state.human_game.make_move(idx, "X")
                    st.session_state.total_games_played += 1
                    
                    # Check if human won
                    if not st.session_state.human_game.winner and not st.session_state.human_game.is_draw():
                        # AI move
                        state = st.session_state.human_game.get_state()
                        available = st.session_state.human_game.available_moves()
                        
                        if available:
                            q_values = st.session_state.agent_o.get_q_values(state)
                            best_move = available[0]
                            best_score = -999
                            
                            for move in available:
                                if q_values[move] > best_score:
                                    best_score = q_values[move]
                                    best_move = move
                            
                            time.sleep(0.5)  # AI thinking delay
                            st.session_state.human_game.make_move(best_move, "O")
                    
                    st.rerun()
    
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    # Game Status
    st.markdown("""<div style="margin: 1.5rem 0;">""", unsafe_allow_html=True)
    if st.session_state.human_game.winner:
        if st.session_state.human_game.winner == "X":
            st.success("🎉 **Congratulations! You Won!**")
        else:
            st.error("🤖 **AI Wins This Round!**")
    elif st.session_state.human_game.is_draw():
        st.warning("🤝 **It's a Draw!**")
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    # Reset button
    col_reset_a, col_reset_b = st.columns(2)
    with col_reset_a:
        if st.button("🔄 New Game", use_container_width=True, key="reset_btn"):
            st.session_state.human_game.reset()
            st.rerun()
    with col_reset_b:
        if st.button("🧹 Clear All", use_container_width=True, key="clear_btn"):
            st.session_state.agent_x = QLearningAgent()
            st.session_state.agent_o = QLearningAgent()
            st.session_state.human_game = TicTacToe()
            st.session_state.stats = {"episodes": 0, "x_wins": 0, "o_wins": 0, "draws": 0}
            st.session_state.training_count = 0
            st.session_state.total_games_played = 0
            st.rerun()
    
    st.markdown("""</div>""", unsafe_allow_html=True)


with col2:
    st.markdown("""<div class="card">""", unsafe_allow_html=True)
    st.markdown("""<h2>📊 AI Training Analytics</h2>""", unsafe_allow_html=True)
    
    stats = st.session_state.stats
    total_episodes = stats["episodes"]
    
    # Metrics Row 1
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric(
            "🎮 Episodes",
            f"{stats['episodes']:,}",
            delta=f"{st.session_state.training_count} training" if st.session_state.training_count > 0 else None
        )
    with metric_cols[1]:
        st.metric(
            "👤 Your Wins",
            stats['x_wins'],
            delta_color="off"
        )
    
    # Metrics Row 2
    metric_cols2 = st.columns(2)
    with metric_cols2[0]:
        st.metric(
            "🤖 AI Wins",
            stats['o_wins'],
            delta_color="off"
        )
    with metric_cols2[1]:
        st.metric(
            "🤝 Draws",
            stats['draws'],
            delta_color="off"
        )
    
    # Win Rate Analysis
    st.markdown("""<h3 style="margin-top: 2rem;">📈 Performance Metrics</h3>""", unsafe_allow_html=True)
    
    if total_episodes > 0:
        x_win_rate = (stats['x_wins'] / total_episodes) * 100 if total_episodes > 0 else 0
        o_win_rate = (stats['o_wins'] / total_episodes) * 100 if total_episodes > 0 else 0
        draw_rate = (stats['draws'] / total_episodes) * 100 if total_episodes > 0 else 0
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <p style="margin-bottom: 0.5rem;"><strong>X Win Rate:</strong> <span style="color: #38bdf8;">{x_win_rate:.1f}%</span></p>
            <div class="progress-container">
                <div class="progress-bar" style="width: {x_win_rate}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <p style="margin-bottom: 0.5rem;"><strong>AI Win Rate:</strong> <span style="color: #38bdf8;">{o_win_rate:.1f}%</span></p>
            <div class="progress-container">
                <div class="progress-bar" style="width: {o_win_rate}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <p style="margin-bottom: 0.5rem;"><strong>Draw Rate:</strong> <span style="color: #38bdf8;">{draw_rate:.1f}%</span></p>
            <div class="progress-container">
                <div class="progress-bar" style="width: {draw_rate}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Learning Concepts
    st.markdown("""<h3>🧠 AI Learning Concepts</h3>""", unsafe_allow_html=True)
    st.markdown("""
    - **Q-Learning**: Agent learns value of each action in each state
    - **Reward System**: +1 for wins, +0.5 for draws, 0 for losses
    - **Exploration vs Exploitation**: Balance between trying new moves and using known good ones
    - **Reinforcement Learning**: Improves strategy through continuous self-play
    """)
    
    st.markdown("""</div>""", unsafe_allow_html=True)

# Footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: #94a3b8; margin-top: 2rem;">
    <p style="font-size: 0.9rem;">
        Made with ❤️ using Streamlit | Q-Learning + Tic Tac Toe AI
    </p>
</div>
""", unsafe_allow_html=True)
