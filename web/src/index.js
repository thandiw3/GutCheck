"""
React frontend for GutCheck web application.

This module provides a React-based frontend for the GutCheck web application,
designed for pathologists, lab technicians, and researchers.
"""

import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
