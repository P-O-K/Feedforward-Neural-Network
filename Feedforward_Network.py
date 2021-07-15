
import numpy as np
import os


class FeedForwardNetwork( object ):
    
    ACTIVATION_METHOD = 'sigmoid'; # = 'tanh'
    MAX_LEARNING_RATE = 1.0;
    MIN_LEARNING_RATE = None;
    LEARNING_RATE = None;
    PROGRESS_PERCENT = 10;

    
    def getLearningRate( self, ERROR ):
        if not self.LEARNING_RATE:
            if self.MAX_LEARNING_RATE:
                if ERROR > self.MAX_LEARNING_RATE: return self.MAX_LEARNING_RATE;
            if self.MIN_LEARNING_RATE:
                if ERROR < self.MIN_LEARNING_RATE: return self.MIN_LEARNING_RATE;
            return ERROR
        return self.LEARNING_RATE
            
    

    def __init__( self, shape:list ):
        super( FeedForwardNetwork, self ).__init__( )
        shapeWeight = [ ( a, b ) for a, b in zip( shape[ 1: ], shape[ :-1 ] ) ]
        self.weights = [ np.random.standard_normal( s )/s[ 1 ]**0.5 for s in shapeWeight ]
        self.biases = [ np.random.standard_normal( ( s, 1 ) ) for s in shape[ 1: ] ]



    def arrayHandle( self, inputArray, labelArray, randomize=True, epoc=1 ):
        if len( inputArray ) != len( labelArray ): return None

        maxIterations = int( len( inputArray ) *epoc )
        for ix in range( maxIterations +1 ):
            IDX = np.random.permutation( len( inputArray ) ) if randomize else np.arange( len( inputArray ) )
            self.runInstance( inputArray[ IDX[ ix %len( IDX ) ] ], labelArray[ IDX[ ix %len( IDX ) ] ] )
            if self.PROGRESS_PERCENT > 0: self.progressReport( ix, maxIterations )



    def runInstance( self, input_data, label_data ):
        datarray = self.feedForward( input_data )
        self.backPropagation( datarray, label_data )



    def feedForward( self, input_data ):
        datarray = [ np.array( input_data ) ]
        for w, b in zip( self.weights, self.biases ):
            datarray.append( self.activate( np.matmul( w, datarray[ -1 ] ) +b ) )
        return datarray



    def backPropagation( self, datarray, label_data ):
        local_error = label_data -datarray[ -1 ]
        deltas = [ local_error *( self.derivative( datarray[ -1 ] ) ) ]

        LR = self.getLearningRate( np.mean( np.abs( local_error ) ) )

        for da, w in zip( datarray[ 1:-1 ][ ::-1 ], self.weights[ 1: ][ ::-1 ] ):
            local_error = np.matmul( w.T, deltas[ -1 ] )
            deltas.append( local_error *self.derivative( da ) )

        for da, de, w, b in zip( datarray[ :-1 ][ ::-1 ], deltas, self.weights[ ::-1 ],	 self.biases[ ::-1 ] ):
            w += np.matmul( de, da.T ) *LR
            b += np.sum( de, axis=0, keepdims=True ) *LR



    def activate( self, x ):
        methods = { 'sigmoid': lambda x: 1 /( 1 +np.exp( -x ) ),
                    'tanh':    lambda x: np.tanh( x ) }
        return methods[ self.ACTIVATION_METHOD ]( x )



    def derivative( self, x ):
        methods = { 'sigmoid': lambda x: x *( 1 -x ),
                    'tanh':    lambda x: 1 - x**2 }
        return methods[ self.ACTIVATION_METHOD ]( x )



    def progressReport( self, arg1, maxIterations ):
        if arg1 %int( np.ceil( ( maxIterations /100 ) *self.PROGRESS_PERCENT ) ) == 0:
            print( f'Progress Update: { arg1 /maxIterations :.1%}' )


    def predictor( self, input_data ):
        return self.feedForward( input_data )[ -1 ]



    def testNetwork( self, testData, testLabels ):
        compareArrays = lambda x, y: ( x == y ).all( )
        amountCorrect = 0
        for i in range( len( testData ) ):
            pred, labl = self.predictor( testData[ i ] ), testLabels[ i ]
            if compareArrays( np.transpose( pred ).round( ), np.transpose( labl ) ): amountCorrect +=1

        amountCorrectPCT = float( amountCorrect ) /len(  testData )
        print( f'TESTED( {len( testData )} )  ->  CORRECT( {amountCorrect} )  ->  ACCURACY( {amountCorrectPCT:.2%} )' )



    def saveWeights( self, fileName='FFN_Weights', location=os.getcwd( ) ):
        saveData = np.append( self.weights, self.biases )
        np.save( location +'\\' +fileName, saveData )



    def loadWeights( self, fileName='FFN_Weights', ext='.npy', location=os.getcwd( ) ):
        filePath = location +'\\' +fileName +ext
        if os.path.isfile( filePath ):
            loadedData = np.load( filePath )
            splitDataLength = len( loadedData )//2 
            self.weights = loadedData[ :splitDataLength ]
            self.biases  = loadedData[ splitDataLength: ]
            return True
        else:
            print( f'File: {filePath} does\'nt exist.' )
            return False
