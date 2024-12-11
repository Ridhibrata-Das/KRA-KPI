import React, { useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Card,
  CardContent,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useDispatch, useSelector } from 'react-redux';
import { fetchKPIs } from '../../redux/slices/kpiSlice';

const KPICard = ({ kpi }) => {
  const getStatusColor = () => {
    if (!kpi.values || kpi.values.length === 0) return 'grey';
    const lastValue = kpi.values[kpi.values.length - 1].value;
    if (lastValue >= kpi.target_value) return 'success.main';
    if (lastValue >= kpi.min_threshold) return 'warning.main';
    return 'error.main';
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {kpi.name}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Target: {kpi.target_value}
          </Typography>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              bgcolor: getStatusColor(),
              ml: 'auto',
            }}
          />
        </Box>
        <ResponsiveContainer width="100%" height={100}>
          <LineChart data={kpi.values || []}>
            <Line
              type="monotone"
              dataKey="value"
              stroke="#8884d8"
              dot={false}
            />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide />
            <Tooltip />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default function KPIDashboard() {
  const dispatch = useDispatch();
  const { kpis, loading } = useSelector((state) => state.kpi);

  useEffect(() => {
    dispatch(fetchKPIs());
  }, [dispatch]);

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '400px',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        KPI Dashboard
      </Typography>
      <Grid container spacing={3}>
        {kpis.map((kpi) => (
          <Grid item xs={12} sm={6} md={4} key={kpi.id}>
            <KPICard kpi={kpi} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
