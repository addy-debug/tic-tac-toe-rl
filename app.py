# app.py
# Q-Learning Tic Tac Toe AI with Streamlit
# Q-Learning + NumPy + Streamlit

import streamlit as st
import numpy as np
import random
import time

st.set_page_config(page_title="Tic Tac Toe AI", layout="wide")

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

st.title("🎮 Tic Tac Toe AI")

col1, col2 = st.columns(2)

with col1:
    st.header("Play Against AI")
    
    # Display board
    board_display = []
    for i in range(3):
        row = st.columns(3)
        for j in range(3):
            idx = i * 3 + j
            with row[j]:
                if st.button(
                    st.session_state.human_game.board[idx] or "⬜",
                    key=f"cell_{idx}",
                    use_container_width=True,
                    disabled=st.session_state.human_game.winner is not None or st.session_state.human_game.board[idx] != " "
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
                            
                            st.session_state.human_game.make_move(best_move, "O")
                    
                    st.rerun()
    
    # Winner display
    if st.session_state.human_game.winner:
        if st.session_state.human_game.winner == "X":
            st.success("🎉 You Win!")
        else:
            st.error("🤖 AI Wins!")
    elif st.session_state.human_game.is_draw():
        st.warning("🤝 Draw!")
    
    # Reset button
    if st.button("Reset Game", use_container_width=True):
        st.session_state.human_game.reset()
        st.rerun()


with col2:
    st.header("📊 AI Training Stats")
    
    stats = st.session_state.stats
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Episodes", stats["episodes"])
        st.metric("X Wins", stats["x_wins"])
    with col_b:
        st.metric("O Wins", stats["o_wins"])
        st.metric("Draws", stats["draws"])
    
    st.subheader("🧠 Learning Concepts")
    st.markdown("""
    - **Q-Learning**: Learns optimal actions through rewards
    - **Reward System**: Positive for wins, neutral for draws
    - **Exploration vs Exploitation**: Balances trying new moves vs using best moves
    - **Reinforcement Learning**: Improves through self-play
    """)
