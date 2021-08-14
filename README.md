# Feedforward-Neural-Network

General Info:

    Currently supported activation methods: 'sigmoid' & 'tanh'
    Learning rate set to 'None' will uses backPropagation error as rate: Nearing ZERO as error decreases.
    Learning rate can have a MAX &or MIN amount fitted to it: (MAX_LEARNING_RATE & MIN_LEARNING_RATE)
    PROGRESS_PERCENT = N: Shows progress report every N%: Set to 0 to ignore report.
    epoc represents how many times an array will be looped over: eg.. len( data )* ?( 0.5, 1.0, 100, 2500 )

How to use:

    Expected input data numpy format: eg.. numpy.zeros( ( 10, 1 ) )

# XOR Problem Example:
        # -> CREATE DATASET & LABELS
        data   = [ [ [0], [0] ], [ [0], [1] ], [ [1], [0] ], [ [1], [1] ] ]
        labels = [      [1],          [0],          [0],          [1] ]

        # -> CREATE NETWORK
        FFN = FeedForwardNetwork( shape = [ 2, 3, 1 ] )
        
        # -> TRAIN NETWORK
        FFN.arrayHandle( data, labels, randomize=True, epoc=2000 )
        
        # -> TEST NETWORK
        FFN.testNetwork( data, labels )
# MNIST Dataset Example:
        # -> LOAD DATAFRAME
        DATAFRAME = np.load( "mnist.npz" )

        # -> CREATE NETWORK       w/ shape = [ 784, 16, 16, 10 ]
        FFN = FeedForwardNetwork( shape=[ len( DATAFRAME['training_images'][ 0 ] ), 16, 16, 10 ] )

        # -> TRAIN NETWORK
        FFN.arrayHandle( DATAFRAME['training_images'], DATAFRAME['training_labels'], randomize=False, epoc=1 )

        # -> TEST NETWORK
        FFN.testNetwork( DATAFRAME['test_images'], DATAFRAME['test_labels'] )
        
        # -> SAVE WEIGHTS
        FFN.saveWeights( 'fileName' )
