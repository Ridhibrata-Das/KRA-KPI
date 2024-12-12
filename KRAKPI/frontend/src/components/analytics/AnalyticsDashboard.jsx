import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Button,
  Alert,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import { useSelector, useDispatch } from 'react-redux';
import { fetchKPIForecast, fetchAnomalies, fetchCorrelations } from '../../redux/slices/analyticsSlice';

const ForecastChart = ({ data, isLoading, error }) => {
  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!data) return null;

  return (
    <Paper sx={{ p: 2, height: 400 }}>
      <Typography variant="h6" gutterBottom>KPI Forecast</Typography>
      <ResponsiveContainer>
        <LineChart data={[...data.historical.values, ...data.forecast.values]}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#8884d8"
            name="Historical"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#82ca9d"
            name="Forecast"
            strokeDasharray="5 5"
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};

const AnomalyChart = ({ data, isLoading, error }) => {
  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!data) return null;

  return (
    <Paper sx={{ p: 2, height: 400 }}>
      <Typography variant="h6" gutterBottom>Anomaly Detection</Typography>
      <ResponsiveContainer>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" name="Time" />
          <YAxis dataKey="value" name="Value" />
          <ZAxis range={[100]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Legend />
          <Scatter
            name="Normal"
            data={data.filter(point => !point.isAnomaly)}
            fill="#8884d8"
          />
          <Scatter
            name="Anomaly"
            data={data.filter(point => point.isAnomaly)}
            fill="#ff0000"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </Paper>
  );
};

const CorrelationMatrix = ({ data, isLoading, error }) => {
  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!data) return null;

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>KPI Correlations</Typography>
      <Box sx={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th></th>
              {Object.keys(data).map(kpi => (
                <th key={kpi}>{kpi}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(data).map(([kpi, correlations]) => (
              <tr key={kpi}>
                <td>{kpi}</td>
                {Object.values(correlations).map((value, i) => (
                  <td
                    key={i}
                    style={{
                      backgroundColor: `rgba(0, 0, 255, ${Math.abs(value)})`,
                      color: Math.abs(value) > 0.5 ? 'white' : 'black',
                    }}
                  >
                    {value.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Box>
    </Paper>
  );
};

const RealTimeMonitoring = ({ kpiId }) => {
  const [alerts, setAlerts] = useState([]);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/kpi/${kpiId}/monitoring`);
    
    ws.onmessage = (event) => {
      const alert = JSON.parse(event.data);
      setAlerts(prev => [...prev, alert].slice(-5)); // Keep last 5 alerts
    };

    return () => ws.close();
  }, [kpiId]);

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>Real-time Alerts</Typography>
      {alerts.map((alert, index) => (
        <Alert
          key={index}
          severity={alert.severity}
          sx={{ mb: 1 }}
        >
          {alert.message}
        </Alert>
      ))}
    </Paper>
  );
};

export default function AnalyticsDashboard() {
  const dispatch = useDispatch();
  const [selectedKPI, setSelectedKPI] = useState('');
  const { kpis } = useSelector(state => state.kpi);
  const {
    forecast,
    anomalies,
    correlations,
    loading,
    error
  } = useSelector(state => state.analytics);

  useEffect(() => {
    if (selectedKPI) {
      dispatch(fetchKPIForecast(selectedKPI));
      dispatch(fetchAnomalies(selectedKPI));
      dispatch(fetchCorrelations([selectedKPI]));
    }
  }, [selectedKPI, dispatch]);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Select KPI</InputLabel>
            <Select
              value={selectedKPI}
              onChange={(e) => setSelectedKPI(e.target.value)}
            >
              {kpis.map(kpi => (
                <MenuItem key={kpi.id} value={kpi.id}>
                  {kpi.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <ForecastChart
            data={forecast}
            isLoading={loading}
            error={error}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <AnomalyChart
            data={anomalies}
            isLoading={loading}
            error={error}
          />
        </Grid>
        
        <Grid item xs={12}>
          <CorrelationMatrix
            data={correlations}
            isLoading={loading}
            error={error}
          />
        </Grid>
        
        <Grid item xs={12}>
          <RealTimeMonitoring kpiId={selectedKPI} />
        </Grid>
      </Grid>
    </Box>
  );
}
