import React, { useState, useEffect } from 'react';
import {
  Container, Typography, Box, Paper, Stepper, Step, StepLabel,
  Button, CircularProgress, Grid, Tabs, Tab, AppBar, Divider,
  Card, CardContent, CardActions, FormControl, InputLabel, Select,
  MenuItem, FormControlLabel, Checkbox, TextField, Snackbar, Alert
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { styled } from '@mui/system';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import BarChartIcon from '@mui/icons-material/BarChart';
import CompareIcon from '@mui/icons-material/Compare';
import BiotechIcon from '@mui/icons-material/Biotech';
import SettingsIcon from '@mui/icons-material/Settings';
import './App.css';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Green - representing microbiome/health
    },
    secondary: {
      main: '#0288d1', // Blue - representing data/science
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
});

// Styled components
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const ResultImage = styled('img')({
  width: '100%',
  maxHeight: '400px',
  objectFit: 'contain',
  marginTop: '16px',
  marginBottom: '16px',
  border: '1px solid #ddd',
  borderRadius: '4px',
});

// Main App component
function App() {
  // State variables
  const [activeTab, setActiveTab] = useState(0);
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Session state
  const [sessionId, setSessionId] = useState(null);
  const [otuFile, setOtuFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  
  // Training parameters
  const [modelType, setModelType] = useState('random_forest');
  const [preprocessing, setPreprocessing] = useState('standard');
  const [featureSelection, setFeatureSelection] = useState(false);
  const [featureSelectionMethod, setFeatureSelectionMethod] = useState('rfe');
  const [nFeatures, setNFeatures] = useState(20);
  const [crossValidation, setCrossValidation] = useState(true);
  const [cvFolds, setCvFolds] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [visualize, setVisualize] = useState(true);
  const [interpret, setInterpret] = useState(false);
  const [tuneHyperparameters, setTuneHyperparameters] = useState(false);
  
  // Results state
  const [trainingResults, setTrainingResults] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);
  const [visualizationResults, setVisualizationResults] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  
  // Multi-class parameters
  const [classType, setClassType] = useState('four');
  const [compareModels, setCompareModels] = useState(false);
  
  // Synthetic data parameters
  const [nSamples, setNSamples] = useState(1000);
  const [nFeaturesGen, setNFeaturesGen] = useState(100);
  const [classBalance, setClassBalance] = useState(0.5);
  const [multiClass, setMultiClass] = useState(false);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setActiveStep(0);
    setError(null);
    setSuccess(null);
  };
  
  // Handle file upload
  const handleFileUpload = async (event) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('otu_file', otuFile);
      if (metadataFile) {
        formData.append('metadata_file', metadataFile);
      }
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to upload files');
      }
      
      setSessionId(data.session_id);
      setSuccess('Files uploaded successfully');
      setActiveStep(1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle model training
  const handleTrainModel = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          otu_file: otuFile.name,
          metadata_file: metadataFile ? metadataFile.name : null,
          model_type: modelType,
          preprocessing: preprocessing,
          feature_selection: featureSelection,
          feature_selection_method: featureSelectionMethod,
          n_features: nFeatures,
          cross_validation: crossValidation,
          cv_folds: cvFolds,
          test_size: testSize,
          visualize: visualize,
          interpret: interpret,
          tune_hyperparameters: tuneHyperparameters,
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to train model');
      }
      
      setTrainingResults(data);
      setSuccess('Model trained successfully');
      setActiveStep(2);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle prediction
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          model_path: trainingResults.model_path,
          otu_file: otuFile.name,
          interpret: interpret,
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to make predictions');
      }
      
      setPredictionResults(data);
      setSuccess('Predictions made successfully');
      setActiveStep(3);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle visualization
  const handleVisualize = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          otu_file: otuFile.name,
          metadata_file: metadataFile ? metadataFile.name : null,
          plot_types: ['diversity', 'abundance', 'heatmap', 'pca'],
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate visualizations');
      }
      
      setVisualizationResults(data);
      setSuccess('Visualizations generated successfully');
      setActiveStep(2);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle model comparison
  const handleCompareModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          otu_file: otuFile.name,
          metadata_file: metadataFile ? metadataFile.name : null,
          models: ['random_forest', 'gradient_boosting', 'svm', 'logistic_regression'],
          preprocessing: preprocessing,
          cross_validation: crossValidation,
          cv_folds: cvFolds,
          test_size: testSize,
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to compare models');
      }
      
      setComparisonResults(data);
      setSuccess('Model comparison completed successfully');
      setActiveStep(2);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle multi-class classification
  const handleMultiClass = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/multiclass', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          otu_file: otuFile.name,
          metadata_file: metadataFile ? metadataFile.name : null,
          class_type: classType,
          model_type: modelType,
          compare: compareModels,
          test_size: testSize,
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to train multi-class model');
      }
      
      setTrainingResults(data);
      setSuccess('Multi-class model trained successfully');
      setActiveStep(2);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle synthetic data generation
  const handleGenerateSyntheticData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          n_samples: nSamples,
          n_features: nFeaturesGen,
          class_balance: classBalance,
          multi_class: multiClass,
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate synthetic data');
      }
      
      setSessionId(data.session_id);
      setOtuFile({ name: data.otu_file });
      setMetadataFile({ name: data.metadata_file });
      setSuccess('Synthetic data generated successfully');
      setActiveStep(1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Render steps for each tab
  const renderSteps = () => {
    switch (activeTab) {
      case 0: // Train
        return ['Upload Data', 'Configure Model', 'Train Model', 'View Results'];
      case 1: // Visualize
        return ['Upload Data', 'Configure Visualizations', 'View Visualizations'];
      case 2: // Compare
        return ['Upload Data', 'Configure Comparison', 'View Comparison Results'];
      case 3: // Multi-class
        return ['Upload Data', 'Configure Multi-class Model', 'View Results'];
      case 4: // Generate
        return ['Configure Synthetic Data', 'Generate Data', 'Use Generated Data'];
      default:
        return ['Step 1', 'Step 2', 'Step 3'];
    }
  };
  
  // Render content for each tab and step
  const renderContent = () => {
    // Common upload step
    if (activeStep === 0 && activeTab !== 4) {
      return (
        <Box sx={{ mt: 4, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Upload your microbiome data files
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="subtitle1" gutterBottom>
                  OTU Data File (Required)
                </Typography>
                <Button
                  component="label"
                  variant="contained"
                  startIcon={<CloudUploadIcon />}
                  sx={{ mt: 2 }}
                >
                  Select OTU File
                  <VisuallyHiddenInput 
                    type="file" 
                    onChange={(e) => setOtuFile(e.target.files[0])}
                  />
                </Button>
                {otuFile && (
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Selected: {otuFile.name}
                  </Typography>
                )}
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="subtitle1" gutterBottom>
                  Metadata File (Optional)
                </Typography>
                <Button
                  component="label"
                  variant="contained"
                  startIcon={<CloudUploadIcon />}
                  sx={{ mt: 2 }}
                >
                  Select Metadata File
                  <VisuallyHiddenInput 
                    type="file" 
                    onChange={(e) => setMetadataFile(e.target.files[0])}
                  />
                </Button>
                {metadataFile && (
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Selected: {metadataFile.name}
                  </Typography>
                )}
              </Paper>
            </Grid>
          </Grid>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleFileUpload}
              disabled={!otuFile || loading}
              sx={{ minWidth: 200 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Upload Files'}
            </Button>
          </Box>
        </Box>
      );
    }
    
    // Tab-specific content
    switch (activeTab) {
      case 0: // Train
        if (activeStep === 1) {
          return (
            <Box sx={{ mt: 4, mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Configure Model Training
              </Typography>
              
              <Paper sx={{ p: 3 }}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Model Type</InputLabel>
                      <Select
                        value={modelType}
                        label="Model Type"
                        onChange={(e) => setModelType(e.target.value)}
                      >
                        <MenuItem value="random_forest">Random Forest</MenuItem>
                        <MenuItem value="gradient_boosting">Gradient Boosting</MenuItem>
                        <MenuItem value="svm">Support Vector Machine</MenuItem>
                        <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
                        <MenuItem value="neural_network">Neural Network</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Preprocessing Method</InputLabel>
                      <Select
                        value={preprocessing}
                        label="Preprocessing Method"
                        onChange={(e) => setPreprocessing(e.target.value)}
                      >
                        <MenuItem value="standard">Standard Scaling</MenuItem>
                        <MenuItem value="robust">Robust Scaling</MenuItem>
                        <MenuItem value="minmax">Min-Max Scaling</MenuItem>
                        <MenuItem value="clr">Centered Log-Ratio (CLR)</MenuItem>
                        <MenuItem value="alr">Additive Log-Ratio (ALR)</
(Content truncated due to size limit. Use line ranges to read in chunks)