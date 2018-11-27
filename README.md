# ResNetTrain
ResNet model achieving 26% test error

network configuration:
 - Adam optimizer with learning rate 0.001
 - two dropout layers after each ReLu activation with drop probability 0.1
 - xavier normal weight initialization
 - flipped the order of skip connection and ReLu activation so that ReLu happens before the skip connection
 - trained up until the 9th epoch
    
