# Imports
from tkinter import *
import shogi

from ..strategy.agents import neural_agent, simple_agent, neural_network

# Local path when dealing with images (not used for now, later when we'll use
# images of the pieces it will come in use)
#ROOT_FOLDER = os.path.dirname(__file__)


# Constants
X_OFFSET = 90
Y_OFFSET = 120
TILE_SIZE = 50

PIECES = {'K' : 'KING', 'L' : 'LANCE', 'R' : 'ROOK', 'P' : 'PAWN', 'G' : 'GOLD', 'N' : 'KNIGHT', 'S' : 'SILVER', 'B' : 'BISHOP'}

JAPANESE_SYMBOLS = {
    'PAWN': '\u6b69',
    'LANCE': '\u9999',
    'KNIGHT': '\u6842',
    'SILVER': '\u9280',
    'GOLD':  '\u91d1',
    'BISHOP':  '\u89d2',
    'ROOK': '\u98db',
    'KING': '\u7389',
}

PIECE_SYMBOLS = {
    1: 'PAWN',
    2: 'LANCE',
    3: 'KNIGHT',
    4: 'SILVER',
    5: 'GOLD',
    6: 'BISHOP',
    7: 'ROOK',
    8: 'KING',
}

USI_NAME = {
    'PAWN': 'P',
    'LANCE': 'L',
    'KNIGHT': 'N',
    'SILVER': 'S',
    'GOLD':  'G',
    'BISHOP':  'B',
    'ROOK': 'R',
    'KING': 'K',
}

# Constant values that we'll use to describe the state of the game
WaitingForAI=1
WaitingForHumanFirstClick=2
WaitingForHumanSecondClick=3
WaitingForHumanResponseToDialog=4


# Function that transforms a Counter object to its List counterpart
def counterToList(a):
    l = []
    for i in a:
        l += [i] * a[i]
    return l




class UI :
    """
    A UI instance. It is a purely virtual board that allows the player to
    challenge the AI in a visual way, by giving him an interface.
    It is coordinated by a controller which uses a simple automaton that
    regulates turns, clicks etc.

    Methods:
    __init__ : initializes a new UI instance.
    play_AI : asks the strategy.py module to find the best move for the AI.
    click : a mouse click event handler for our interface.
    display_board : a function that draws the elements of the interface on the canvas.
    """


    def __init__(self, board, agent):
        """
        Initializes a new UI instance.

        Parameters:
        board (shogi.Board): the board to give to the AI algorithm

        Returns:
        None
        """

        self.agent = agent

        # We'll need this when we'll be using images instead of text
        # self.images = []

        # Store the board somewhere and update it at each move
        self.board = board

        # The initial state of the game is 'waiting for human to select a piece'
        self.state = WaitingForHumanFirstClick
        self.selected = None
        self.move = None

        # We instantiate a Tkinter window
        master = Tk()

        # We create a proprely-sized canvas
        self.canvas = Canvas(master, width=9*TILE_SIZE+2*X_OFFSET, height=9*TILE_SIZE+2*Y_OFFSET, background="#ffeab0")

        # We add a listener to the canvas so that it uses mouse clicks
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.pack()

        # Aaaand we can finally display the board and open the window !
        self.display_board()
        mainloop()



    def play_AI(self) :
        """
        Summary or Description of the Function

        Parameters:
        None

        Returns:
        move (shogi.Move): the move chosen by the AI based on its evaluation
        """

        move = self.agent.play(self.board)

        self.board.push(move)

        # Then we update the board and the game state
        self.display_board()
        self.state = WaitingForHumanFirstClick

        # Return the selected move
        return move


    def click(self, event):
        """
        Listener for the canevas.
        Update the board and the game state depending on the position of
        the click event.

        Parameters:
        event (Tkinter.Event): the click event on our canvas, with it position

        Returns:
        None
        """

        # If the dialog window is open, we'll check whether the click position
        # corresponds to one of the buttons (YES/NO)
        if (self.state == WaitingForHumanResponseToDialog) :

            if ( 3 < (event.x-X_OFFSET)/TILE_SIZE < 4.42 and 4.7 < (event.y-Y_OFFSET)/TILE_SIZE < 5.2) :
                # A click on "YES"
                self.board.push(shogi.Move.from_usi(self.move + "+"))
                self.state = WaitingForAI
                self.display_board()
                self.play_AI()

            elif ( 4.58 < (event.x-X_OFFSET)/TILE_SIZE < 6 and 4.7 < (event.y-Y_OFFSET)/TILE_SIZE < 5.2):
                # A click on "NO"
                self.board.push(shogi.Move.from_usi(self.move))
                self.state = WaitingForAI
                self.display_board()
                self.play_AI()

        # Otherwise, it means it has to be a click on a board square
        else :

            # Compute the line and column
            ligne = int((event.y-Y_OFFSET)/TILE_SIZE)
            colonne = int((event.x-X_OFFSET)/TILE_SIZE)

            # If we are still waiting for the human player's first click
            if self.state == WaitingForHumanFirstClick:

                # If the line is n°9, it corresponds to pieces in hand
                if ligne == 9 :
                    l = counterToList(self.board.pieces_in_hand[0])

                    # Check that he did click on a piece in particular
                    if colonne < len(l) :
                        self.selected = USI_NAME[PIECE_SYMBOLS[l[colonne]]] + '*'
                        # Update the game state
                        self.state = WaitingForHumanSecondClick

                # Otherwise, that means that he clicked on one of the board's squares
                elif ( 0 <= ligne < 9 and 0 <= colonne < 9) :
                    self.selected = str(9-colonne)+"abcdefghi"[ligne]
                    self.state = WaitingForHumanSecondClick

            # If we are waiting for the human's second click (piece already selected)
            elif self.state == WaitingForHumanSecondClick :

                # Check that the click is indeed on the board
                if (0 <= ligne < 9 and 0 <= colonne < 9) :
                    move = self.selected +str(9-colonne) +"abcdefghi"[ligne]

                    # Check that it is a legal move
                    if shogi.Move.from_usi(move) in self.board.legal_moves:

                        # Conditions for promoting :
                        # The piece is not yet in its promoted state
                        # The piece is not being dropped
                        # The piece enters, leaves, or moves inside the promotion zone

                        if move[0] in "123456789" :
                            # Check that the piece isn't being dropped

                            if ligne < 3 or move[1] in "abc" :
                                # Check that the move starts or ends in the promotion zone

                                position = (9-int(move[0])) + "abcdefghi".index(move[1])*9

                                if self.board.piece_at(position).symbol() in "SLNPRB" and not self.board.piece_at(position).is_promoted():
                                    # Check that the piece is not already promoted
                                    self.state = WaitingForHumanResponseToDialog
                                    self.move = move

                        # Otherwise, play the move as usual
                        if self.state != WaitingForHumanResponseToDialog :
                            self.board.push(shogi.Move.from_usi(move))
                            self.display_board()
                            self.state = WaitingForAI
                            self.play_AI()

                    # Otherwise, deselect the selected piece and go back to waiting for human's first click
                    else :
                        self.state = WaitingForHumanFirstClick
                        self.selected=None

                # Otherwise, deselect the selected piece and go back to waiting for human's first click
                else :
                    self.state = WaitingForHumanFirstClick
                    self.selected=None


        # When all of this is done, update the canvas
        self.display_board()




    def display_board(self):
        """
        Draws the board on the canvas.

        Parameters:
        None

        Returns:
        None
        """

        # Empty the canvas
        self.canvas.delete("all")

        # Draw the board
        for i in range(0, 10) :
            # Horizontal lines
            self.canvas.create_line(X_OFFSET, TILE_SIZE*i+Y_OFFSET, X_OFFSET+TILE_SIZE*9, TILE_SIZE*i+Y_OFFSET)
            # Vertical lines
            self.canvas.create_line(TILE_SIZE*i+X_OFFSET, Y_OFFSET, TILE_SIZE*i+X_OFFSET, Y_OFFSET+9*TILE_SIZE)


        # For each case, draw the piece that goes on it
        for ligne in range(9):
            for colonne in range(9):
                a = self.board.piece_at(ligne*9+colonne)

                # If there's a piece on this case
                if a != None:
                    x = X_OFFSET + TILE_SIZE*colonne + TILE_SIZE/2
                    y = Y_OFFSET + TILE_SIZE*ligne + TILE_SIZE/2

                    # Draw it in red if selected, black otherwise
                    if self.selected == str(9-colonne) + "abcdefghi"[ligne] :
                        color='red'
                    elif a.color==0:
                        color='black'
                    else :
                        color='gray'

                    # Draw it on the canvas
                    self.canvas.create_text(x,y,font=("CODE2000", int(TILE_SIZE/3)), fill=color,text=a.japanese_symbol())


        # Draw the pieces in human's hands
        l = self.board.pieces_in_hand[0]
        c = 0
        for i in l :
            color='black'
            if self.selected != None and self.selected[0] in "KLRPGNSB" and PIECES[self.selected[0]] == PIECE_SYMBOLS[i] :
                color='red'
            x = X_OFFSET + c*TILE_SIZE + 1/2*TILE_SIZE
            y = Y_OFFSET + 9 * TILE_SIZE + 1/2*TILE_SIZE
            self.canvas.create_text(x- TILE_SIZE/6,y,font=("CODE2000", int(TILE_SIZE/3)), fill=color,text=JAPANESE_SYMBOLS[PIECE_SYMBOLS[i]])
            self.canvas.create_text(x + TILE_SIZE/4,y+TILE_SIZE/3,font=("CODE2000", int(TILE_SIZE/6)), fill=color,text=str(l[i]))
            c += 1

        """
        for i in range(len(l)) :
            x = X_OFFSET + i*TILE_SIZE + 1/2*TILE_SIZE
            y = Y_OFFSET + 9 * TILE_SIZE + 1/2*TILE_SIZE
            self.canvas.create_text(x,y,font=("CODE2000", int(TILE_SIZE/3)), fill=color,text=JAPANESE_SYMBOLS[PIECE_SYMBOLS[l[i]]])
        """

        # Draw the pieces in IA's hands
        l = self.board.pieces_in_hand[1]
        c = 0
        for i in l :
            color='gray'
            x = X_OFFSET + c*TILE_SIZE + 1/2*TILE_SIZE
            y = Y_OFFSET - 1/2*TILE_SIZE
            # Draw the piece itself
            self.canvas.create_text(x- TILE_SIZE/6,y,font=("CODE2000", int(TILE_SIZE/3)), fill=color,text=JAPANESE_SYMBOLS[PIECE_SYMBOLS[i]])
            # Write how many pieces of this type he owns
            self.canvas.create_text(x + TILE_SIZE/4,y+TILE_SIZE/3,font=("CODE2000", int(TILE_SIZE/6)), fill=color,text=str(l[i]))
            c += 1


        # Draw the "promotion dialog"
        if self.state==WaitingForHumanResponseToDialog :

            # Window
            self.canvas.create_rectangle(X_OFFSET+2.5*TILE_SIZE, Y_OFFSET+3.5*TILE_SIZE, X_OFFSET+6.5*TILE_SIZE, Y_OFFSET+5.5*TILE_SIZE, fill="white")

            # "Promotion ?"
            self.canvas.create_text(X_OFFSET+4.5*TILE_SIZE, Y_OFFSET+4.1*TILE_SIZE, font=("CODE2000", int(TILE_SIZE/4)), fill='black', text="PROMOTE ?")

            # "YES" button
            self.canvas.create_rectangle(X_OFFSET+3*TILE_SIZE, Y_OFFSET+4.7*TILE_SIZE, X_OFFSET+4.42*TILE_SIZE, Y_OFFSET+5.2*TILE_SIZE, fill="white")
            self.canvas.create_text(X_OFFSET+3.7*TILE_SIZE, Y_OFFSET+4.97*TILE_SIZE, font=("CODE2000", int(TILE_SIZE/4)), fill='black', text="YES")

            # "NO" button
            self.canvas.create_rectangle(X_OFFSET+4.58*TILE_SIZE, Y_OFFSET+4.7*TILE_SIZE, X_OFFSET+6*TILE_SIZE, Y_OFFSET+5.2*TILE_SIZE, fill="white")
            self.canvas.create_text(X_OFFSET+5.3*TILE_SIZE, Y_OFFSET+4.97*TILE_SIZE, font=("CODE2000", int(TILE_SIZE/4)), fill='black', text="NO")

        # Update the canvas
        self.canvas.update_idletasks()



# Create a new UI instance
if __name__=="__main__" :

    # Creation of a board
    board = shogi.Board()
    board.reset()

    # Creation of a Keras model and a NeuralAgent
    model = neural_network.create_model()
    n_a = neural_agent.NeuralAgent(model)

    # Creation of a simple agent
    s_a = simple_agent.SimpleAgent()

    # Launch the UI with that board
    #a = UI(board, s_a)
    a = UI(board, n_a)
