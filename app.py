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
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    /* Hide hamburger menu and header */
    header {
        display: none !important;
    }
    
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Hide the three bars menu */
    .stToolbar {
        display: none !important;
    }
    
    /* Header Styling */
    .stMarkdown h1 {
        color: #00d4ff !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
        letter-spacing: 2px !important;
    }
    
    .stMarkdown h2 {
        color: #00d4ff !important;
        font-size: 1.8rem !important;
        margin-top: 0 !important;
        margin-bottom: 1.5rem !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3) !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown h3 {
        color: #00d4ff !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Card Container */
    .card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.8) 0%, rgba(26, 26, 46, 0.8) 100%) !important;
        border: 2px solid #00d4ff !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        margin-bottom: 0 !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Button Base Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: #0a0e27 !important;
        border: 2px solid #00d4ff !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0099cc 0%, #0077aa 100%) !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Success/Error/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 212, 100, 0.15) 0%, rgba(0, 180, 80, 0.15) 100%) !important;
        border: 2px solid #00d464 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: #00ff99 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 100, 100, 0.15) 0%, rgba(220, 50, 50, 0.15) 100%) !important;
        border: 2px solid #ff6464 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: #ff9999 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(100, 180, 255, 0.15) 0%, rgba(80, 150, 220, 0.15) 100%) !important;
        border: 2px solid #64b4ff !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: #99ccff !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    /* Text Colors */
    p, li {
        color: #e0e0e0 !important;
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
    }
    
    .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Column Containers */
    [data-testid="column"] {
        background: transparent !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0, 212, 255, 0.2) !important;
        width: 60% !important;
        margin-left: auto !important;
        margin-right: auto !important;
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
    st.session_state.training_count = 0


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

            # DRAW
            elif training_game.is_draw():
                reward = 0.5

            # UPDATE Q TABLE
            if current_player == "X":
                st.session_state.agent_x.update_q_value(state, action, reward, next_state)
            else:
                st.session_state.agent_o.update_q_value(state, action, reward, next_state)

            # END GAME
            if training_game.winner or training_game.is_draw():
                st.session_state.training_count += 1
                break

            # SWITCH PLAYER
            current_player = "O" if current_player == "X" else "X"


# Train AI in background
if st.session_state.training_count < 1000:
    train_ai(10)


# =====================================================
# UI
# =====================================================

st.markdown("""<h1>🎮 TIC TAC TOE AI 🤖</h1>""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""<div class="card"><h2>⚡ PLAY AGAINST AI</h2>""", unsafe_allow_html=True)
    
    # Game board with simple layout
    board_cols = []
    for row_idx in range(3):
        row = st.columns(3)
        board_cols.append(row)
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            with board_cols[i][j]:
                cell_value = st.session_state.human_game.board[idx]
                
                # Display X and O
                if cell_value == "X":
                    display_text = "❌"
                    button_disabled = True
                elif cell_value == "O":
                    display_text = "⭕"
                    button_disabled = True
                else:
                    display_text = "⬜"
                    button_disabled = st.session_state.human_game.winner is not None
                
                if st.button(
                    display_text,
                    key=f"cell_{idx}",
                    use_container_width=True,
                    disabled=button_disabled
                ):
                    # Human move
                    st.session_state.human_game.make_move(idx, "X")
                    
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
    
    # Game Status
    st.markdown("""<br>""", unsafe_allow_html=True)
    if st.session_state.human_game.winner:
        if st.session_state.human_game.winner == "X":
            st.success("🎉 YOU WON! CONGRATULATIONS!")
        else:
            st.error("🤖 AI WINS THIS ROUND!")
    elif st.session_state.human_game.is_draw():
        st.warning("🤝 IT'S A DRAW!")
    
    st.markdown("""<br>""", unsafe_allow_html=True)
    
    # Buttons
    col_reset_a, col_reset_b = st.columns(2)
    with col_reset_a:
        if st.button("🔄 NEW GAME", use_container_width=True, key="reset_btn"):
            st.session_state.human_game.reset()
            st.rerun()
    with col_reset_b:
        if st.button("🧹 RESET AI", use_container_width=True, key="clear_btn"):
            st.session_state.agent_x = QLearningAgent()
            st.session_state.agent_o = QLearningAgent()
            st.session_state.human_game = TicTacToe()
            st.session_state.training_count = 0
            st.rerun()
    
    st.markdown("""</div>""", unsafe_allow_html=True)


with col2:
    st.markdown("""<div class="card"><h2>🧠 HOW IT WORKS</h2>""", unsafe_allow_html=True)
    
    st.markdown("""
    ### Q-Learning AI
    
    The AI uses **Q-Learning**, a reinforcement learning algorithm that learns from self-play.
    
    #### Key Concepts:
    
    **Q-Values**: The AI maintains a table of values for each possible board state and move combination. Higher Q-values indicate better moves.
    
    **Rewards**: 
    - ✅ Win: +1.0
    - 🤝 Draw: +0.5
    - ❌ Loss: 0.0
    
    **Exploration vs Exploitation**: 
    - The AI explores new moves randomly (20% chance)
    - It exploits best known moves (80% chance)
    - This balance helps find optimal strategies
    
    **Learning Rate (α)**: 0.1 - How quickly the AI updates its knowledge
    
    **Discount Factor (γ)**: 0.95 - How much the AI values future rewards
    
    #### Training Progress:
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <p style="font-size: 1.3rem; color: #00d4ff; font-weight: bold;">
    🎯 Training Episodes: <span style="color: #00ff99;">{st.session_state.training_count}</span>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    
    ### Tips to Beat the AI:
    
    1. **Corner Strategy** - Start in corners (0, 2, 6, 8)
    2. **Center Control** - Control the center (position 4)
    3. **Block Threats** - Always block the AI's winning moves
    4. **Create Forks** - Try to create two winning opportunities
    
    ### About This Project:
    
    This is a demonstration of reinforcement learning in action. The AI improves every time you play!
    
    """, unsafe_allow_html=True)
    
    st.markdown("""</div>""", unsafe_allow_html=True)

# Footer
st.markdown("""<hr>""", unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; color: #00d4ff; font-weight: bold; font-size: 1.1rem; margin-top: 2rem;">
Made with ❤️ | Q-Learning + Tic Tac Toe | Reinforcement Learning Demo
</p>
""", unsafe_allow_html=True)
