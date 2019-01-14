library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

keras::backend()$clear_session()

classes <- c( "background", "tumor" )
numberOfClassificationLabels <- length( classes )
segmentationLabelWeights <- c( 1.0, 2.0 )
resampledImageSize <- c( 112, 160, 112 )
numberOfFiltersAtBaseLayer <- 32

# imageMods <- c( "T1", "T1c", "T2", "Flair" )
imageMods <- c( "T2", "Flair" )
channelSize <- length( imageMods )
batchSize <- 16L
segmentationLabels <- seq_len( numberOfClassificationLabels ) - 1

baseDirectory <- '/home/nick/Data/BrainTumor/'
scriptsDirectory <- paste0( baseDirectory, '/Scripts/WholeTumorSegmentation/' )
source( paste0( scriptsDirectory, 'unetBatchGenerator.R' ) )

templateDirectory <- paste0( baseDirectory, 'Template/' )
reorientTemplateDirectory <- templateDirectory
reorientTemplate <- antsImageRead( paste0( reorientTemplateDirectory, "S_template3.nii.gz" ) )

dataDirectory <- paste0( baseDirectory, 'Nifti_brats_tcia/' )

brainImageFiles <- list.files( path = dataDirectory,
  pattern = "T1.nii.gz", full.names = TRUE, recursive = TRUE )

trainingImageFiles <- list()
trainingMaskFiles <- list()
trainingSegmentationFiles <- list()
trainingTransforms <- list()

missingFiles <- c()

cat( "Loading data...\n" )
pb <- txtProgressBar( min = 0, max = length( brainImageFiles ), style = 3 )

count <- 1
for( i in seq_len( length( brainImageFiles ) ) )
  {
  setTxtProgressBar( pb, i )

  subjectDirectory <- dirname( brainImageFiles[i] )

  brainSegmentationFile <- ''
  brainMaskFile <- ''
  brainImageFile <- ''
  fwdtransforms <- c()
  invtransforms <- c()

  brainImageFile <- brainImageFiles[i]
  brainMaskFile <- sub( "T1", "BrainExtractionMask", brainImageFile )
  brainSegmentationFile <- sub( "T1", "segmentationWholeTumor", brainImageFile )

  multiModalityFiles <- c()
  for( j in seq_len( channelSize ) )
    {
    multiModalityFiles[j] <- sub( "T1", imageMods[j], brainImageFile )
    if( ! file.exists( multiModalityFiles[j] ) )
      {
      stop( "File ", multiModalityFiles[j], " does not exist.\n" )
      }
    }

  xfrmPrefix <- "kirbyxT1"
  xfrmFiles <- list.files( subjectDirectory, pattern = paste0( xfrmPrefix, "*" ), full.names = TRUE )

  fwdtransforms[1] <- xfrmFiles[2]                    # FALSE
  fwdtransforms[2] <- xfrmFiles[1]                    # TRUE

  invtransforms[1] <- xfrmFiles[1]                    # FALSE
  invtransforms[2] <- xfrmFiles[3]                    # FALSE

  missingFile <- FALSE
  for( j in seq_len( length( fwdtransforms ) ) )
    {
    if( !file.exists( invtransforms[j] ) || !file.exists( fwdtransforms[j] ) )
      {
      missingFile <- TRUE
      }
    }

  if( ! file.exists( brainImageFile ) )
    {
    missingFile <- TRUE
    }

  if( ! file.exists( brainMaskFile ) )
    {
    missingFile <- TRUE
    }

  if( missingFile )
    {
    missingFiles <- append( missingFiles, subjectDirectory )
    } else {
    trainingTransforms[[count]] <- list(
      fwdtransforms = fwdtransforms, invtransforms = invtransforms )

    trainingImageFiles[[count]] <- multiModalityFiles
    trainingSegmentationFiles[[count]] <- brainSegmentationFile
    trainingMaskFiles[[count]] <- brainMaskFile

    count <- count + 1
    }
  }
cat( "\n" )


###
#
# Create the Unet model
#

# See this thread:  https://github.com/rstudio/tensorflow/issues/272

with( tf$device( "/cpu:0" ), {
unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFiltersAtBaseLayer, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )
  } )

unetModel %>% compile( loss = "categorical_crossentropy",
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = c( "acc" ) )

if( file.exists( paste0( scriptsDirectory, "/wholeTumorSegmentationWeights.h5" ) ) )
  {
  load_model_weights_hdf5( unetModel, paste0( scriptsDirectory, "/wholeTumorSegmentationWeights.h5" ) )
  }




parallel_unetModel <- multi_gpu_model( unetModel, gpus = 8 )


# parallel_unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
#   optimizer = optimizer_adam( lr = 0.0001 ),
#   metrics = c( multilabel_dice_coefficient ) )

# parallel_unetModel %>% compile( loss = "categorical_crossentropy",
#   optimizer = optimizer_adam( lr = 0.0001 ),
#   metrics = c( "acc" ) )




###
#
# Set up the training generator
#

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfData )
validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )


###
#
# Run training
#

# See comments here
# https://github.com/keras-team/keras/issues/3653#issuecomment-405849928
# for imbalanced data

track <- unetModel %>% fit_generator(
  generator = unetImageBatchGenerator( batchSize = batchSize,
                                       resampledImageSize = resampledImageSize,
                                       segmentationLabels = segmentationLabels,
                                       segmentationLabelWeights = segmentationLabelWeights,
                                       doRandomHistogramMatching = FALSE,
                                       reorientImage = reorientTemplate,
                                       sourceImageList = trainingImageFiles[trainingIndices],
                                       segmentationList = trainingSegmentationFiles[trainingIndices],
                                       sourceMaskList = trainingMaskFiles[trainingIndices],
                                       sourceTransformList = trainingTransforms[trainingIndices],
                                       outputFile = paste0( scriptsDirectory, "trainingData.csv" )
                                     ),
  steps_per_epoch = 24L,
  epochs = 100,
  validation_data = unetImageBatchGenerator( batchSize = batchSize,
                                       resampledImageSize = resampledImageSize,
                                       segmentationLabels = segmentationLabels,
                                       segmentationLabelWeights = segmentationLabelWeights,
                                       doRandomHistogramMatching = FALSE,
                                       reorientImage = reorientTemplate,
                                       sourceImageList = trainingImageFiles[validationIndices],
                                       sourceMaskList = trainingMaskFiles[validationIndices],
                                       segmentationList = trainingSegmentationFiles[validationIndices],
                                       sourceTransformList = trainingTransforms[validationIndices],
                                       outputFile = paste0( scriptsDirectory, "validationData.csv" )
                                     ),
  validation_steps = 8L,
  callbacks = list(
    callback_model_checkpoint( paste0( scriptsDirectory, "/wholeTumorSegmentationWeights.h5" ),
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
       patience = 10 )
  )
)

save_model_weights_hdf5( unetModel, paste0( scriptsDirectory, "/wholeTumorSegmentationWeights.h5" ) )

