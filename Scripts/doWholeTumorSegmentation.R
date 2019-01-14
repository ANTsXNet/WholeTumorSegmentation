library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

# imageMods <- c( "T1", "T1c", "T2", "Flair" )
imageMods <- c( "T2", "Flair" )
channelSize <- length( imageMods )

if( length( args ) != 3 + channelSize )
  {
  helpMessage <- paste0( "Usage:  Rscript doWholeTumorSegmentation.R",
    " inputMod1File inputMod2File ... inputMaskFile outputFilePrefix reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  index <- 1

  inputFileNames <- c()
  for( i in seq_len( channelSize ) )
    {
    inputFileNames[index] <- args[index]
    index <- index + 1
    }
  inputMaskFileName <- args[index]
  index <- index + 1
  outputFilePrefix <- args [index]
  index <- index + 1
  reorientTemplateFileName <- args[index]
  }

classes <- c( "background", "tumor" )
numberOfClassificationLabels <- length( classes )
segmentationLabelWeights <- c( 1.0, 2.0 )
resampledImageSize <- c( 112, 160, 112 )
numberOfFiltersAtBaseLayer <- 32

batchSize <- 16L

numberOfClassificationLabels <- length( classes )
segmentationLabelWeights <- c( 1.0, 1000.0, 1000.0 )
resampledImageSize <- c( 112, 160, 112 )
numberOfFiltersAtBaseLayer <- 32
segmentationLabels <- seq_len( numberOfClassificationLabels ) - 1

cat( "Reading reorientation template", reorientTemplateFileName )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFiltersAtBaseLayer, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- getPretrainedNetwork( "wholeTumorSegmentationT2Flair" )
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading input." )
startTime <- Sys.time()

mask <- antsImageRead( inputMaskFileName, dimension = 3 )

images <- list()
for( i in seq_len( channelSize ) )
  {
  images[[i]] <- antsImageRead( inputFileNames[i], dimension = 3 )
  images[[i]][mask < 0.5] <- 0
  }
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template and cropping to mask." )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( images[[1]] )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )

warpedMask <- applyAntsrTransformToImage( xfrm, mask, reorientTemplate,
  interpolation = "nearestNeighbor" )
warpedMask <- iMath( warpedMask, "MD", 3 )

batchX <- array( data = 0,
  dim = c( 1, resampledImageSize, channelSize ) )

warpedCroppedImages <- list()
for( i in seq_len( channelSize ) )
  {
  warpedImage <- applyAntsrTransformToImage( xfrm, images[[i]], reorientTemplate )
  warpedCroppedImages[[i]] <- cropImage( warpedImage, warpedMask, 1 )
  if( i == 1 )
    {
    originalCroppedSize <- dim( warpedCroppedImages[[1]] )
    }
  warpedCroppedImages[[i]] <- resampleImage( warpedCroppedImages[[i]],
    resampledImageSize, useVoxels = TRUE )
  warpedCroppedImages[[i]] <-
    ( warpedCroppedImages[[i]] - mean( warpedCroppedImages[[i]] ) ) /
      sd( warpedCroppedImages[[i]] )

  batchX[1,,,, i] <- as.array( warpedCroppedImages[[i]] )
  }
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, warpedCroppedImages[[1]] )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Renormalize to native space" )
startTime <- Sys.time()

zeroArray <- array( data = 0, dim = dim( warpedMask ) )
zeroImage <- as.antsImage( zeroArray, reference = warpedMask )

probabilityImageTmp <- resampleImage( probabilityImagesArray[[1]][[1]],
  originalCroppedSize, useVoxels = TRUE )
probabilityImageTmp <- decropImage( probabilityImageTmp, zeroImage )
probabilityImageBackground <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImageTmp, mask )

probabilityImageTmp <- resampleImage( probabilityImagesArray[[1]][[2]],
  originalCroppedSize, useVoxels = TRUE )
probabilityImageTmp <- decropImage( probabilityImageTmp, zeroImage )
probabilityImageTumor <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImageTmp, mask )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Writing", outputFilePrefix )
startTime <- Sys.time()
antsImageWrite( probabilityImageBackground, paste0( outputFilePrefix, "Background.nii.gz" ) )
antsImageWrite( probabilityImageTumor, paste0( outputFilePrefix, "Tumor.nii.gz" ) )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
