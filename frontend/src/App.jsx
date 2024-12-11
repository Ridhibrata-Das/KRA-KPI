import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { Provider } from 'react-redux';
import theme from './theme';
import store from './redux/store';

// Components
import Login from './components/auth/Login';
import KPIDashboard from './components/dashboard/KPIDashboard';
import KPICreationForm from './components/KPICreationForm';
import PrivateRoute from './components/auth/PrivateRoute';

function App() {
  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route
              path="/dashboard"
              element={
                <PrivateRoute>
                  <KPIDashboard />
                </PrivateRoute>
              }
            />
            <Route
              path="/kpi/create"
              element={
                <PrivateRoute>
                  <KPICreationForm />
                </PrivateRoute>
              }
            />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Router>
      </ThemeProvider>
    </Provider>
  );
}

export default App;
