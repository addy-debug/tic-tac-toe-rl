# app.py
# Error-Free Self Playing Tic Tac Toe AI
# Q-Learning + Flask + NumPy

from flask import Flask, jsonify, request, render_template_string
import numpy as np
import random
import threading
import time

app = Flask(__name__)

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
# GLOBALS
# =====================================================

agent_x = QLearningAgent()
agent_o = QLearningAgent()

training_game = TicTacToe()
human_game = TicTacToe()

stats = {
    "episodes": 0,
    "x_wins": 0,
    "o_wins": 0,
    "draws": 0
}

training_running = True


# =====================================================
# AI TRAINING
# =====================================================

def train_ai():

    global training_running

    while training_running:

        try:

            training_game.reset()

            current_player = "X"

            while True:

                state = training_game.get_state()

                available = training_game.available_moves()

                if current_player == "X":
                    action = agent_x.choose_action(state, available)
                else:
                    action = agent_o.choose_action(state, available)

                training_game.make_move(action, current_player)

                next_state = training_game.get_state()

                reward = 0

                # WIN
                if training_game.winner == current_player:

                    reward = 1

                    if current_player == "X":
                        stats["x_wins"] += 1
                    else:
                        stats["o_wins"] += 1

                # DRAW
                elif training_game.is_draw():

                    reward = 0.5
                    stats["draws"] += 1

                # UPDATE Q TABLE
                if current_player == "X":
                    agent_x.update_q_value(
                        state,
                        action,
                        reward,
                        next_state
                    )
                else:
                    agent_o.update_q_value(
                        state,
                        action,
                        reward,
                        next_state
                    )

                # END GAME
                if training_game.winner or training_game.is_draw():
                    break

                # SWITCH PLAYER
                current_player = "O" if current_player == "X" else "X"

            stats["episodes"] += 1

            # Prevent CPU overload
            time.sleep(0.001)

        except Exception as e:
            print("Training Error:", e)


# Start Training Thread
training_thread = threading.Thread(target=train_ai)
training_thread.daemon = True
training_thread.start()


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/move", methods=["POST"])
def move():

    data = request.get_json()

    position = data.get("position")

    # INVALID CLICK
    if human_game.board[position] != " ":

        return jsonify({
            "board": human_game.board,
            "winner": None
        })

    # HUMAN MOVE
    human_game.make_move(position, "X")

    # HUMAN WIN
    if human_game.winner:

        return jsonify({
            "board": human_game.board,
            "winner": "Human"
        })

    # DRAW
    if human_game.is_draw():

        return jsonify({
            "board": human_game.board,
            "winner": "Draw"
        })

    # AI MOVE
    state = human_game.get_state()

    available = human_game.available_moves()

    q_values = agent_o.get_q_values(state)

    best_move = available[0]
    best_score = -999

    for move in available:

        if q_values[move] > best_score:
            best_score = q_values[move]
            best_move = move

    human_game.make_move(best_move, "O")

    # AI WIN
    if human_game.winner:

        return jsonify({
            "board": human_game.board,
            "winner": "AI"
        })

    # DRAW
    if human_game.is_draw():

        return jsonify({
            "board": human_game.board,
            "winner": "Draw"
        })

    return jsonify({
        "board": human_game.board,
        "winner": None
    })


@app.route("/reset")
def reset():

    human_game.reset()

    return jsonify({
        "status": "reset"
    })


@app.route("/stats")
def get_stats():

    return jsonify(stats)


# =====================================================
# FRONTEND
# =====================================================

HTML_PAGE = """

<!DOCTYPE html>

<html>

<head>

    <title>Tic Tac Toe AI</title>

    <style>

        *{
            margin:0;
            padding:0;
            box-sizing:border-box;
        }

        body{

            font-family:Arial;
            background:#0f172a;
            color:white;

            display:flex;
            justify-content:center;
            align-items:center;

            min-height:100vh;
        }

        .container{

            width:950px;

            display:grid;

            grid-template-columns:1fr 1fr;

            gap:30px;
        }

        .card{

            background:#1e293b;

            padding:30px;

            border-radius:20px;

            box-shadow:0 0 20px rgba(0,0,0,0.3);
        }

        h1{

            margin-bottom:20px;

            color:#38bdf8;
        }

        .board{

            display:grid;

            grid-template-columns:repeat(3, 110px);

            gap:10px;

            justify-content:center;
        }

        .cell{

            width:110px;
            height:110px;

            border:none;

            border-radius:15px;

            background:#334155;

            color:white;

            font-size:40px;

            cursor:pointer;

            transition:0.2s;
        }

        .cell:hover{

            transform:scale(1.05);

            background:#475569;
        }

        .winner{

            margin-top:20px;

            font-size:24px;

            color:#4ade80;
        }

        .stats{

            line-height:2;

            font-size:18px;
        }

        button.reset{

            margin-top:20px;

            border:none;

            padding:12px 20px;

            border-radius:10px;

            background:#38bdf8;

            color:white;

            cursor:pointer;

            font-size:16px;
        }

        button.reset:hover{

            background:#0ea5e9;
        }

    </style>

</head>

<body>

<div class="container">

    <div class="card">

        <h1>Play Against AI</h1>

        <div class="board" id="board"></div>

        <div class="winner" id="winner"></div>

        <button class="reset" onclick="resetGame()">
            Reset Game
        </button>

    </div>

    <div class="card">

        <h1>AI Training Stats</h1>

        <div class="stats">

            <p>Episodes: <span id="episodes">0</span></p>

            <p>X Wins: <span id="xwins">0</span></p>

            <p>O Wins: <span id="owins">0</span></p>

            <p>Draws: <span id="draws">0</span></p>

        </div>

        <div style="margin-top:30px;">

            <h2>Learning Concepts</h2>

            <ul style="margin-top:10px; line-height:2;">

                <li>Q Learning</li>
                <li>Reward System</li>
                <li>Exploration vs Exploitation</li>
                <li>Reinforcement Learning</li>

            </ul>

        </div>

    </div>

</div>

<script>

let board = document.getElementById("board");

let winnerText = document.getElementById("winner");

let cells = [];

function createBoard(){

    board.innerHTML = "";

    cells = [];

    for(let i = 0; i < 9; i++){

        let cell = document.createElement("button");

        cell.classList.add("cell");

        cell.innerText = "";

        cell.onclick = () => {

            if(cell.innerText === ""){
                makeMove(i);
            }
        };

        board.appendChild(cell);

        cells.push(cell);
    }
}

async function makeMove(position){

    const response = await fetch("/move", {

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({
            position:position
        })
    });

    const data = await response.json();

    updateBoard(data.board);

    if(data.winner){

        if(data.winner === "Draw"){
            winnerText.innerText = "Game Draw";
        }
        else{
            winnerText.innerText = data.winner + " Wins!";
        }
    }
}

function updateBoard(boardData){

    for(let i = 0; i < 9; i++){

        cells[i].innerText = boardData[i];
    }
}

async function resetGame(){

    await fetch("/reset");

    winnerText.innerText = "";

    createBoard();
}

async function updateStats(){

    const response = await fetch("/stats");

    const data = await response.json();

    document.getElementById("episodes").innerText = data.episodes;

    document.getElementById("xwins").innerText = data.x_wins;

    document.getElementById("owins").innerText = data.o_wins;

    document.getElementById("draws").innerText = data.draws;
}

setInterval(updateStats, 500);

createBoard();

</script>

</body>

</html>

"""


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False,
        port=5000
    )
