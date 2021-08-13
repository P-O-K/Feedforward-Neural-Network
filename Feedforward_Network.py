
import numpy as np
import os


class FeedforwardNetwork( object ):
    
    ACTIVATION_METHOD :str   = 'sigmoid';
    MAX_LEARNING_RATE :float = 1.0;
    MIN_LEARNING_RATE :float = None;
    LEARNING_RATE     :float = None;
    PROGRESS_PERCENT  :int   = 10;
    

    def __init__( self, shape:list ) -> None:
        super( FeedforwardNetwork, self ).__init__( )
        shapeWeight = [ ( a, b ) for a, b in zip( shape[ 1: ], shape[ :-1 ] ) ]
        self.weights = [ np.random.standard_normal( s )/s[ 1 ]**0.5 for s in shapeWeight ]
        self.biases = [ np.random.standard_normal( ( s, 1 ) ) for s in shape[ 1: ] ]



    def arrayHandle( self, inputArray:list, labelArray:list, randomize:bool=True, epoc:int=1 ) -> None:
        if len( inputArray ) != len( labelArray ): return None

        maxIterations = int( len( inputArray ) *epoc )
        for ix in range( maxIterations +1 ):
            IDX = np.random.permutation( len( inputArray ) ) if randomize else np.arange( len( inputArray ) )
            self.runInstance( inputArray[ IDX[ ix %len( IDX ) ] ], labelArray[ IDX[ ix %len( IDX ) ] ] )
            if self.PROGRESS_PERCENT > 0: self.progressReport( ix, maxIterations, self.PROGRESS_PERCENT )



    def runInstance( self, input_data:list, label_data:list ) -> None:
        datarray = self.feedForward( input_data )
        self.backPropagation( datarray, label_data )



    def feedForward( self, input_data:list ) -> np.array:
        datarray = [ np.array( input_data ) ]
        for w, b in zip( self.weights, self.biases ):
            datarray.append( self.activate( np.matmul( w, datarray[ -1 ] ) +b, self.ACTIVATION_METHOD ) )
        return datarray



    def backPropagation( self, datarray:np.array, label_data:np.array ) -> None:
        local_error = label_data -datarray[ -1 ]
        deltas = [ local_error *( self.derivative( datarray[ -1 ], self.ACTIVATION_METHOD ) ) ]
        
        LR = self.getLearningRate( self.LEARNING_RATE, self.MAX_LEARNING_RATE, self.MIN_LEARNING_RATE, np.mean( np.abs( local_error ) ) )

        for da, w in zip( datarray[ 1:-1 ][ ::-1 ], self.weights[ 1: ][ ::-1 ] ):
            local_error = np.matmul( w.T, deltas[ -1 ] )
            deltas.append( local_error *self.derivative( da, self.ACTIVATION_METHOD ) )

        for da, de, w, b in zip( datarray[ :-1 ][ ::-1 ], deltas, self.weights[ ::-1 ],  self.biases[ ::-1 ] ):
            w += np.matmul( de, da.T ) *LR
            b += np.sum( de, axis=0, keepdims=True ) *LR



    @staticmethod
    def activate( x:np.array, A_TYPE:str ) -> np.array:
        methods = { 'sigmoid': lambda x: 1 /( 1 +np.exp( -x ) ),
                    'tanh':    lambda x: np.tanh( x ) }
        return methods[ A_TYPE ]( x )



    @staticmethod
    def derivative( x:np.array, A_TYPE:str ) -> np.array:
        methods = { 'sigmoid': lambda x: x *( 1 -x ),
                    'tanh':    lambda x: 1 - x**2 }
        return methods[ A_TYPE ]( x )



    @staticmethod
    def getLearningRate( LR:float, MXLR:float, MNLR:float, ERROR:float ) -> float:
        if not LR:
            if MXLR and ERROR > MXLR: return MXLR;
            if MNLR and ERROR < MNLR: return MNLR;
            return ERROR
        return LR



    @staticmethod
    def progressReport( arg1, maxIterations:int, percent:int ) -> None:
        if arg1 %int( np.ceil( ( maxIterations /100 ) *percent ) ) == 0:
            print( f'Progress Update: { arg1 /maxIterations :.1%}' )



    def predictor( self, input_data:list ) -> np.array:
        return self.feedForward( input_data )[ -1 ]



    def testNetwork( self, testData:np.array, testLabels:np.array ) -> None:
        print( 'Running Test...' )
        compareArrays = lambda x, y: ( x == y ).all( )
        amountCorrect = 0
        for i in range( len( testData ) ):
            pred, labl = self.predictor( testData[ i ] ), testLabels[ i ]
            if compareArrays( np.transpose( pred ).round( ), np.transpose( labl ) ): amountCorrect +=1

        amountCorrectPCT = float( amountCorrect ) /len(  testData )
        print( f'TESTED( {len( testData )} )  ->  CORRECT( {amountCorrect} )  ->  ACCURACY( {amountCorrectPCT:.2%} )' )



    def saveWeights( self, fileName:str='FFN_Weights.npy', location:str=os.getcwd( ) ) -> None:
        saveData = np.append( self.weights, self.biases )
        np.save( location +'\\' +fileName, saveData )



    def loadWeights( self, fileName:str='FFN_Weights.npy', location:str=os.getcwd( ) ) -> bool:
        filePath = location +'\\' +fileName
        if os.path.isfile( filePath ):
            loadedData = np.load( filePath )
            splitDataLength = len( loadedData )//2 
            self.weights = loadedData[ :splitDataLength ]
            self.biases  = loadedData[ splitDataLength: ]
            return True
        else:
            print( f'File: {filePath} does\'nt exist.' )
            return False


if __name__ == '__main__':
    # -> LOAD DATAFRAME
    DATAFRAME = np.load( "mnist.npz" )

    # -> CREATE NETWORK       w/ shape = [ 784, 16, 16, 10 ]
    FFN = FeedforwardNetwork( shape=[ len( DATAFRAME['training_images'][ 0 ] ), 16, 16, 10 ] )

    # -> TRAIN NETWORK
    FFN.arrayHandle( DATAFRAME['training_images'], DATAFRAME['training_labels'], randomize=False, epoc=1 )

    # -> TEST NETWORK
    FFN.testNetwork( DATAFRAME['test_images'], DATAFRAME['test_labels'] )
