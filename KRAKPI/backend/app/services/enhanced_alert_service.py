from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..models.alert import Alert, AlertRule, AlertNotification, AlertSeverity, AlertChannel
from ..models.kpi import KPI, KPIValue
import asyncio
import json
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
from twilio.rest import Client
import jinja2
import logging

class NotificationManager:
    def __init__(self, config: Dict):
        self.config = config
        self.email_template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates/emails')
        )
        self.twilio_client = Client(
            config['twilio_account_sid'],
            config['twilio_auth_token']
        ) if 'twilio_account_sid' in config else None

    async def send_email(self, recipient: str, subject: str, template_name: str, context: Dict) -> bool:
        try:
            template = self.email_template_env.get_template(template_name)
            html_content = template.render(**context)
            
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = self.config['smtp_from_email']
            message['To'] = recipient
            
            message.attach(MIMEText(html_content, 'html'))
            
            await aiosmtplib.send(
                message,
                hostname=self.config['smtp_host'],
                port=self.config['smtp_port'],
                username=self.config['smtp_username'],
                password=self.config['smtp_password'],
                use_tls=True
            )
            return True
        except Exception as e:
            logging.error(f"Email sending failed: {str(e)}")
            return False

    async def send_sms(self, phone_number: str, message: str) -> bool:
        try:
            if not self.twilio_client:
                raise ValueError("Twilio not configured")
            
            message = self.twilio_client.messages.create(
                body=message,
                from_=self.config['twilio_from_number'],
                to=phone_number
            )
            return True
        except Exception as e:
            logging.error(f"SMS sending failed: {str(e)}")
            return False

    async def send_slack(self, webhook_url: str, payload: Dict) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logging.error(f"Slack notification failed: {str(e)}")
            return False

class AlertEvaluator:
    def __init__(self, db: Session):
        self.db = db

    def evaluate_threshold_condition(self, value: float, params: Dict) -> bool:
        operator = params.get('operator')
        threshold = params.get('value')
        
        if operator == 'gt':
            return value > threshold
        elif operator == 'lt':
            return value < threshold
        elif operator == 'gte':
            return value >= threshold
        elif operator == 'lte':
            return value <= threshold
        return False

    def evaluate_trend_condition(self, values: List[float], params: Dict) -> bool:
        window_size = params.get('window_size', 5)
        trend_type = params.get('trend_type')
        threshold = params.get('threshold', 0.1)
        
        if len(values) < window_size:
            return False
        
        recent_values = values[-window_size:]
        slope = (recent_values[-1] - recent_values[0]) / (window_size - 1)
        
        if trend_type == 'increasing':
            return slope > threshold
        elif trend_type == 'decreasing':
            return slope < -threshold
        return False

    def evaluate_pattern_condition(self, values: List[float], params: Dict) -> bool:
        pattern_type = params.get('pattern_type')
        window_size = params.get('window_size', 5)
        
        if len(values) < window_size:
            return False
        
        recent_values = values[-window_size:]
        
        if pattern_type == 'zigzag':
            differences = [recent_values[i+1] - recent_values[i] for i in range(len(recent_values)-1)]
            sign_changes = sum(1 for i in range(len(differences)-1) if differences[i] * differences[i+1] < 0)
            return sign_changes >= params.get('min_changes', 2)
        
        elif pattern_type == 'stable':
            std_dev = statistics.stdev(recent_values)
            return std_dev <= params.get('max_std_dev', 0.1)
        
        return False

class EnhancedAlertService:
    def __init__(self, db: Session, config: Dict):
        self.db = db
        self.notification_manager = NotificationManager(config)
        self.evaluator = AlertEvaluator(db)
        self.alert_cache = {}  # Cache for cooldown management

    async def process_kpi_value(self, kpi_id: int, value: float, timestamp: datetime):
        """Process a new KPI value and trigger alerts if necessary"""
        rules = self.db.query(AlertRule).filter(
            AlertRule.kpi_id == kpi_id,
            AlertRule.is_active == True
        ).all()

        for rule in rules:
            # Check cooldown
            cache_key = f"{kpi_id}_{rule.id}"
            if cache_key in self.alert_cache:
                last_alert_time = self.alert_cache[cache_key]
                if datetime.utcnow() - last_alert_time < timedelta(minutes=rule.cooldown_minutes):
                    continue

            should_alert = False
            if rule.condition_type == 'threshold':
                should_alert = self.evaluator.evaluate_threshold_condition(
                    value, rule.condition_params
                )
            elif rule.condition_type == 'trend':
                recent_values = self.get_recent_values(kpi_id)
                should_alert = self.evaluator.evaluate_trend_condition(
                    recent_values, rule.condition_params
                )
            elif rule.condition_type == 'pattern':
                recent_values = self.get_recent_values(kpi_id)
                should_alert = self.evaluator.evaluate_pattern_condition(
                    recent_values, rule.condition_params
                )

            if should_alert:
                await self.create_and_send_alert(rule, value, timestamp)
                self.alert_cache[cache_key] = datetime.utcnow()

    async def create_and_send_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Create an alert and send notifications"""
        alert = Alert(
            rule_id=rule.id,
            kpi_id=rule.kpi_id,
            severity=rule.severity,
            message=self.generate_alert_message(rule, value),
            triggered_value=value,
            details={
                'condition_type': rule.condition_type,
                'condition_params': rule.condition_params,
                'timestamp': timestamp.isoformat()
            }
        )
        self.db.add(alert)
        self.db.commit()

        # Send notifications through configured channels
        for channel in rule.channels:
            if channel == AlertChannel.EMAIL:
                for recipient in rule.condition_params.get('email_recipients', []):
                    success = await self.notification_manager.send_email(
                        recipient,
                        f"KPI Alert: {rule.name}",
                        'alert_email.html',
                        {'alert': alert, 'rule': rule}
                    )
                    self.record_notification(alert.id, channel, recipient, success)

            elif channel == AlertChannel.SMS:
                for recipient in rule.condition_params.get('sms_recipients', []):
                    success = await self.notification_manager.send_sms(
                        recipient,
                        self.generate_alert_message(rule, value)
                    )
                    self.record_notification(alert.id, channel, recipient, success)

            elif channel == AlertChannel.SLACK:
                webhook_url = rule.condition_params.get('slack_webhook_url')
                if webhook_url:
                    success = await self.notification_manager.send_slack(
                        webhook_url,
                        self.generate_slack_payload(alert, rule)
                    )
                    self.record_notification(alert.id, channel, webhook_url, success)

    def generate_alert_message(self, rule: AlertRule, value: float) -> str:
        kpi = self.db.query(KPI).get(rule.kpi_id)
        return f"Alert: {rule.name} - {kpi.name} value ({value}) {self.get_condition_description(rule)}"

    def get_condition_description(self, rule: AlertRule) -> str:
        if rule.condition_type == 'threshold':
            operator = rule.condition_params.get('operator')
            threshold = rule.condition_params.get('value')
            return f"{'exceeded' if operator in ['gt', 'gte'] else 'fell below'} threshold {threshold}"
        elif rule.condition_type == 'trend':
            return f"showed {rule.condition_params.get('trend_type')} trend"
        elif rule.condition_type == 'pattern':
            return f"matched {rule.condition_params.get('pattern_type')} pattern"
        return ""

    def generate_slack_payload(self, alert: Alert, rule: AlertRule) -> Dict:
        kpi = self.db.query(KPI).get(rule.kpi_id)
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 KPI Alert: {rule.name}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*KPI:*\n{kpi.name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{alert.severity}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Value:*\n{alert.triggered_value}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
        }

    def record_notification(self, alert_id: int, channel: AlertChannel, recipient: str, success: bool):
        """Record notification attempt in database"""
        notification = AlertNotification(
            alert_id=alert_id,
            channel=channel,
            recipient=recipient,
            status='delivered' if success else 'failed',
            error_message=None if success else "Delivery failed"
        )
        self.db.add(notification)
        self.db.commit()

    def get_recent_values(self, kpi_id: int, hours: int = 24) -> List[float]:
        """Get recent values for a KPI"""
        values = self.db.query(KPIValue).filter(
            KPIValue.kpi_id == kpi_id,
            KPIValue.timestamp >= datetime.utcnow() - timedelta(hours=hours)
        ).order_by(KPIValue.timestamp).all()
        return [v.value for v in values]
