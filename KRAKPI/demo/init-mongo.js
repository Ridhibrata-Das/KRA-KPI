db = db.getSiblingDB('kpi_demo');

// Create demo users
db.users.insertMany([
    {
        email: "demo.admin@kpi-system.com",
        password: "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNZZ6I/9Lx.Ka", // demo_admin_123
        role: "admin",
        name: "Demo Admin",
        created_at: new Date(),
        is_demo: true
    },
    {
        email: "demo.user@kpi-system.com",
        password: "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNZZ6I/9Lx.Ka", // demo_user_123
        role: "user",
        name: "Demo User",
        created_at: new Date(),
        is_demo: true
    }
]);

// Create sample KPIs
db.kpis.insertMany([
    {
        name: "Monthly Sales",
        description: "Total monthly sales in USD",
        target: 100000,
        unit: "USD",
        frequency: "monthly",
        owner_id: db.users.findOne({email: "demo.admin@kpi-system.com"})._id,
        created_at: new Date(),
        is_demo: true
    },
    {
        name: "Customer Satisfaction",
        description: "Average customer satisfaction score",
        target: 4.5,
        unit: "score",
        frequency: "daily",
        owner_id: db.users.findOne({email: "demo.admin@kpi-system.com"})._id,
        created_at: new Date(),
        is_demo: true
    },
    {
        name: "Website Conversion Rate",
        description: "Percentage of visitors who make a purchase",
        target: 2.5,
        unit: "percentage",
        frequency: "daily",
        owner_id: db.users.findOne({email: "demo.user@kpi-system.com"})._id,
        created_at: new Date(),
        is_demo: true
    }
]);

// Generate sample KPI data
let kpis = db.kpis.find().toArray();
let now = new Date();

kpis.forEach(kpi => {
    let dataPoints = [];
    for (let i = 90; i >= 0; i--) {
        let date = new Date(now);
        date.setDate(date.getDate() - i);
        
        let value;
        switch(kpi.name) {
            case "Monthly Sales":
                value = 95000 + Math.random() * 10000;
                break;
            case "Customer Satisfaction":
                value = 4.2 + Math.random() * 0.6;
                break;
            case "Website Conversion Rate":
                value = 2.0 + Math.random() * 1.0;
                break;
        }
        
        dataPoints.push({
            kpi_id: kpi._id,
            value: value,
            timestamp: date,
            is_demo: true
        });
    }
    db.kpi_data.insertMany(dataPoints);
});

// Create sample alerts
db.alerts.insertMany([
    {
        kpi_id: db.kpis.findOne({name: "Monthly Sales"})._id,
        condition: "threshold",
        threshold: 90000,
        operator: "less_than",
        notification_channel: "email",
        owner_id: db.users.findOne({email: "demo.admin@kpi-system.com"})._id,
        created_at: new Date(),
        is_demo: true
    },
    {
        kpi_id: db.kpis.findOne({name: "Customer Satisfaction"})._id,
        condition: "threshold",
        threshold: 4.0,
        operator: "less_than",
        notification_channel: "email",
        owner_id: db.users.findOne({email: "demo.admin@kpi-system.com"})._id,
        created_at: new Date(),
        is_demo: true
    }
]);
