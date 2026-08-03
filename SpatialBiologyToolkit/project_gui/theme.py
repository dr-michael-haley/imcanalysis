"""Shared visual language for the lightweight SBT Project Console."""

from __future__ import annotations


COLORS = {
    "ink": "#241B2F",
    "muted": "#685F72",
    "surface": "#FFFFFF",
    "canvas": "#F6F3F9",
    "border": "#DDD6E5",
    "purple": "#431176",
    "purple_mid": "#7650A8",
    "purple_soft": "#F0EAF7",
    "lavender": "#966BC7",
    "gold": "#F2A900",
    "gold_soft": "#FFF4D6",
    "green": "#1F7A4D",
    "green_soft": "#E7F5ED",
    "red": "#A02C3A",
    "red_soft": "#FBEAEC",
    "blue": "#2F5EA8",
    "blue_soft": "#EAF1FC",
}


APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {COLORS["canvas"]};
    color: {COLORS["ink"]};
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 10pt;
}}

QLabel#pageTitle {{
    color: {COLORS["purple"]};
    font-size: 22pt;
    font-weight: 700;
    padding: 0 0 2px 0;
}}
QLabel#pageSubtitle, QLabel#mutedText {{
    color: {COLORS["muted"]};
    font-size: 10pt;
}}
QLabel#sectionNotice {{
    background: {COLORS["gold_soft"]};
    border: 1px solid #E8C563;
    border-radius: 6px;
    color: #5F4600;
    padding: 8px 10px;
}}
QLabel#configSummary {{
    background: {COLORS["purple_soft"]};
    border: 1px solid #CDBBE0;
    border-radius: 6px;
    color: {COLORS["purple"]};
    font-weight: 600;
    padding: 8px 10px;
}}
QLabel#safetyBadge {{
    background: #321052;
    border: 1px solid #7650A8;
    border-radius: 6px;
    color: #F8F3FC;
    font-size: 8.5pt;
    padding: 8px;
}}
QLabel#resultReady {{
    background: {COLORS["green_soft"]};
    border: 1px solid #8CC9A7;
    border-radius: 6px;
    color: {COLORS["green"]};
    font-size: 12pt;
    font-weight: 700;
    padding: 8px 12px;
}}
QLabel#resultBlocked {{
    background: {COLORS["red_soft"]};
    border: 1px solid #D99DA5;
    border-radius: 6px;
    color: {COLORS["red"]};
    font-size: 12pt;
    font-weight: 700;
    padding: 8px 12px;
}}

QFrame#projectBar, QFrame#metricCard {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 7px;
}}
QFrame#projectBar {{
    border-left: 4px solid {COLORS["gold"]};
}}
QFrame#metricCard QLabel#metricValue {{
    color: {COLORS["purple"]};
    font-size: 18pt;
    font-weight: 700;
}}
QFrame#metricCard QLabel#metricLabel {{
    color: {COLORS["muted"]};
    font-size: 9pt;
}}

QFrame#sidebar {{
    background: {COLORS["purple"]};
    border: none;
}}
QFrame#sidebar QLabel {{
    background: transparent;
    color: white;
}}
QFrame#sidebar QLabel#brandLogo {{
    background: white;
    border: 1px solid #CDBBE0;
    border-radius: 8px;
    padding: 6px;
}}
QListWidget#navigation {{
    background: {COLORS["purple"]};
    border: none;
    color: #F4EDF9;
    font-size: 10.5pt;
    outline: none;
    padding: 3px;
}}
QListWidget#navigation::item {{
    border-radius: 5px;
    margin: 2px 3px;
    padding: 10px 9px;
}}
QListWidget#navigation::item:hover {{
    background: #5B288A;
}}
QListWidget#navigation::item:selected {{
    background: {COLORS["gold"]};
    color: #2B1C00;
    font-weight: 700;
}}

QPushButton {{
    background: {COLORS["surface"]};
    border: 1px solid #BEB4C8;
    border-radius: 5px;
    min-height: 24px;
    padding: 4px 10px;
}}
QPushButton:hover {{
    background: {COLORS["purple_soft"]};
    border-color: {COLORS["purple_mid"]};
}}
QPushButton:pressed {{
    background: #E3D8EE;
}}
QPushButton:disabled {{
    background: #F0EDF2;
    border-color: #D8D2DC;
    color: #9A929F;
}}
QPushButton[role="primary"] {{
    background: {COLORS["purple"]};
    border-color: {COLORS["purple"]};
    color: white;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{
    background: #5A268D;
}}
QPushButton[role="warning"] {{
    background: {COLORS["gold_soft"]};
    border-color: #DDB651;
    color: #5F4600;
}}

QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox,
QDoubleSpinBox, QListWidget, QTableWidget {{
    background: {COLORS["surface"]};
    border: 1px solid #CFC7D7;
    border-radius: 4px;
    selection-background-color: {COLORS["purple_mid"]};
    selection-color: white;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 26px;
    padding: 2px 6px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QListWidget:focus, QTableWidget:focus {{
    border: 2px solid {COLORS["purple_mid"]};
}}
QTextBrowser {{
    padding: 10px;
}}

QHeaderView::section {{
    background: #EAE3F1;
    border: none;
    border-right: 1px solid #D1C7DC;
    border-bottom: 1px solid #C8BDD3;
    color: {COLORS["purple"]};
    font-weight: 700;
    padding: 7px;
}}
QTableWidget {{
    alternate-background-color: #FAF8FC;
    gridline-color: #E7E1EB;
}}
QTableWidget::item {{
    padding: 4px;
}}
QTableWidget::item:selected {{
    background: #D8C7E8;
    color: {COLORS["ink"]};
}}

QGroupBox {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    font-size: 11pt;
    font-weight: 700;
    margin-top: 12px;
    padding-top: 10px;
}}
QGroupBox::title {{
    color: {COLORS["purple"]};
    left: 10px;
    padding: 0 5px;
    subcontrol-origin: margin;
}}

QFrame#configField {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-left: 5px solid #B8AFBF;
    border-radius: 5px;
}}
QFrame#configField[configState="stored"] {{
    background: {COLORS["blue_soft"]};
    border-left-color: {COLORS["blue"]};
}}
QFrame#configField[configState="staged"] {{
    background: {COLORS["gold_soft"]};
    border-left-color: {COLORS["gold"]};
}}
QFrame#configField[configState="pending-reset"] {{
    background: #F8EEF4;
    border-left-color: #B54A7B;
}}
QLabel#fieldName {{
    color: {COLORS["ink"]};
    font-size: 11pt;
    font-weight: 700;
}}
QLabel#stateBadge {{
    border-radius: 8px;
    font-size: 8.5pt;
    font-weight: 700;
    padding: 2px 7px;
}}
QLabel#stateBadge[configState="inherited"] {{
    background: #E8E4EB;
    color: #5E5664;
}}
QLabel#stateBadge[configState="stored"] {{
    background: #D9E6FA;
    color: #234F92;
}}
QLabel#stateBadge[configState="staged"] {{
    background: #F8D77F;
    color: #5B4000;
}}
QLabel#stateBadge[configState="pending-reset"] {{
    background: #EBC9D9;
    color: #7D2750;
}}
QLabel#defaultValue {{
    color: {COLORS["muted"]};
    font-size: 9pt;
}}
QLabel#errorText {{
    color: {COLORS["red"]};
    font-weight: 600;
}}

QTabWidget::pane {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
}}
QTabBar::tab {{
    background: #E9E3EE;
    border: 1px solid #CEC4D6;
    padding: 7px 13px;
}}
QTabBar::tab:selected {{
    background: {COLORS["surface"]};
    color: {COLORS["purple"]};
    font-weight: 700;
}}
QStatusBar {{
    background: #2E0C51;
    color: white;
}}
"""


__all__ = ["APP_STYLESHEET", "COLORS"]
