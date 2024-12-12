"""API Documentation descriptions and examples."""

# Main API description
API_DESCRIPTION = """
# KPI Management System API

## Overview
This API provides comprehensive functionality for managing Key Performance Indicators (KPIs) in an enterprise environment. 
It supports creation, monitoring, analysis, and reporting of KPIs with advanced features like ML-powered predictions and 
real-time alerts.

## Features
* 📊 KPI Creation and Management
* 🎯 Target Setting and Tracking
* 👥 Role-based Access Control
* 📈 Performance Analytics
* 🔔 Real-time Alerts
* 📱 Multi-channel Notifications
* 📋 Comprehensive Reporting

## Authentication
All API endpoints require authentication using JWT (JSON Web Tokens). Include the token in the Authorization header:
```
Authorization: Bearer your_jwt_token
```

## Rate Limiting
API requests are limited to:
* 100 requests per minute for regular endpoints
* 1000 requests per day for analytics endpoints

## Error Handling
The API uses standard HTTP status codes and returns detailed error messages in JSON format:
```json
{
    "detail": "Error description",
    "code": "ERROR_CODE",
    "timestamp": "2024-12-12T22:35:53+05:30"
}
```
"""

# Tags descriptions
TAGS_METADATA = [
    {
        "name": "KPIs",
        "description": "Operations for managing KPI definitions and configurations.",
    },
    {
        "name": "Assignments",
        "description": "Manage KPI assignments to teams, projects, and users.",
    },
    {
        "name": "Analytics",
        "description": "Advanced analytics and ML-powered predictions for KPIs.",
    },
    {
        "name": "Alerts",
        "description": "Configure and manage KPI alerts and notifications.",
    },
    {
        "name": "Reports",
        "description": "Generate and manage KPI reports.",
    },
]

# Example responses
EXAMPLE_KPI = {
    "id": "kpi-123",
    "name": "Quarterly Sales Target",
    "description": "Track quarterly sales performance against targets",
    "type": "CURRENCY",
    "thresholds": {
        "min_value": 0,
        "max_value": 1000000,
        "target_value": 500000,
        "warning_threshold": 0.8,
        "critical_threshold": 0.6
    },
    "time_config": {
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-03-31T23:59:59"
    },
    "assignment": {
        "team_assignments": {
            "sales_team_1": {
                "id": "assignment-123",
                "type": "TEAM",
                "role": "OWNER",
                "permissions": {
                    "can_view": True,
                    "can_edit": True,
                    "can_delete": True,
                    "can_assign": True,
                    "can_approve": True
                },
                "notifications": {
                    "preference": "DAILY",
                    "threshold_alerts": True,
                    "progress_updates": True
                }
            }
        }
    },
    "metadata": {
        "created_by": "user-123",
        "created_at": "2024-12-12T22:35:53+05:30",
        "version": 1
    }
}

EXAMPLE_ASSIGNMENT = {
    "id": "assignment-123",
    "type": "TEAM",
    "target_id": "sales_team_1",
    "role": "OWNER",
    "permissions": {
        "can_view": True,
        "can_edit": True,
        "can_delete": True,
        "can_assign": True,
        "can_approve": True
    },
    "notifications": {
        "preference": "DAILY",
        "threshold_alerts": True,
        "progress_updates": True
    },
    "valid_from": "2024-01-01T00:00:00",
    "valid_until": "2024-12-31T23:59:59",
    "delegation_allowed": True
}

# Operation descriptions
OPERATIONS = {
    "create_kpi": {
        "summary": "Create a new KPI",
        "description": """
        Create a new Key Performance Indicator (KPI) with detailed configuration.
        
        ## Configuration Options
        * Name and description
        * Type (numeric, percentage, currency, etc.)
        * Thresholds and targets
        * Time period
        * Initial assignments
        
        ## Required Permissions
        * `kpi:create`
        
        ## Notes
        * KPI names must be unique within their scope
        * All threshold values must be consistent
        * Time period must be valid
        """,
        "responses": {
            201: {"description": "KPI created successfully"},
            400: {"description": "Invalid input"},
            401: {"description": "Unauthorized"},
            403: {"description": "Insufficient permissions"}
        }
    },
    "update_kpi": {
        "summary": "Update an existing KPI",
        "description": """
        Update configuration of an existing KPI.
        
        ## Updatable Fields
        * Name and description
        * Thresholds and targets
        * Time period
        * Status
        
        ## Required Permissions
        * `kpi:edit`
        * Must be owner or have edit permissions
        
        ## Notes
        * Updates are versioned
        * Previous values are preserved in history
        * Notifications sent to stakeholders
        """,
        "responses": {
            200: {"description": "KPI updated successfully"},
            400: {"description": "Invalid input"},
            401: {"description": "Unauthorized"},
            403: {"description": "Insufficient permissions"},
            404: {"description": "KPI not found"}
        }
    },
    "create_assignment": {
        "summary": "Create KPI assignment",
        "description": """
        Assign a KPI to teams, projects, or users with detailed configuration.
        
        ## Assignment Options
        * Type (team, project, user, department, business unit)
        * Role (owner, contributor, viewer, approver)
        * Permissions
        * Notification preferences
        * Validity period
        
        ## Required Permissions
        * `kpi:assign`
        * Must be owner or have assignment permissions
        
        ## Notes
        * Assignments can be time-limited
        * Multiple assignments allowed
        * Notifications sent to assignees
        """,
        "responses": {
            201: {"description": "Assignment created successfully"},
            400: {"description": "Invalid input"},
            401: {"description": "Unauthorized"},
            403: {"description": "Insufficient permissions"},
            404: {"description": "KPI not found"}
        }
    }
}

# Response descriptions
RESPONSES = {
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Invalid input data",
                    "code": "INVALID_INPUT",
                    "fields": {
                        "name": "Name is required",
                        "thresholds.max_value": "Must be greater than min_value"
                    }
                }
            }
        }
    },
    401: {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Invalid or expired token",
                    "code": "INVALID_TOKEN"
                }
            }
        }
    },
    403: {
        "description": "Forbidden",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Insufficient permissions",
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "required": ["kpi:edit"],
                    "provided": ["kpi:view"]
                }
            }
        }
    },
    404: {
        "description": "Not Found",
        "content": {
            "application/json": {
                "example": {
                    "detail": "KPI not found",
                    "code": "KPI_NOT_FOUND"
                }
            }
        }
    },
    409: {
        "description": "Conflict",
        "content": {
            "application/json": {
                "example": {
                    "detail": "KPI with this name already exists",
                    "code": "KPI_NAME_EXISTS"
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Internal server error",
                    "code": "INTERNAL_ERROR",
                    "trace_id": "abc123"
                }
            }
        }
    }
}
