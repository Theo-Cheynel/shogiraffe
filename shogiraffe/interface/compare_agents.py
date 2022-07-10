import shogi

from ..strategy.agents import neural_agent, random_agent, neural_network


def compare_agents(agent_1, agent_2, nb_games):
    """
    Compares two agents by making them play one against the other.
    """

    # Three variables to keep track of the outcomes of the games
    results_1 = 0
    results_2 = 0
    draw = 0
    average_length = 0

    for i in range(nb_games) :

        print("Now starting game n°"+str(i+1))

        # Play a game until either a draw or a loss occurs.

        b = shogi.Board()

        while True:

            # Play agent_1's move
            move = agent_1.play(b)
            b.push(move)

            # Check for stalemate
            if b.is_stalemate():
                draw += 1
                break

            # Check for checkmate
            if b.is_checkmate():
                results_2 += 1
                break

            # Play agent_2's move
            move = agent_2.play(b)
            b.push(move)

            # Check for stalemate
            if b.is_stalemate():
                draw += 1
                break

            # Check for checkmate
            if b.is_checkmate():
                results_1 += 1
                break

        average_length += b.move_number/nb_games

    print("Results :")
    print("Agent 1 won", results_1, "times.")
    print("Agent 2 won", results_2, "times.")
    print("Number of draws : ", str(draw)+".")
    print("Average game length :", )



# Tests
if __name__ == "__main__":

    # Compare two RandomAgents
    #compare_agents(random_agent.RandomAgent(), random_agent.RandomAgent(), 20)

    # Compare a RandomAgent and a NeuralAgent
    # Warning : currently, a NeuralAgent can only be placed as second player !
    model = neural_network.create_model()
    agent = neural_agent.NeuralAgent(model)
    compare_agents(random_agent.RandomAgent(), agent, 1)