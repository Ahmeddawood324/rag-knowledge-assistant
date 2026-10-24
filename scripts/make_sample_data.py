"""Generate the sample knowledge base (2 PDFs + 2 sheets exported as CSV).

The institute is FICTIONAL. Replace the files in data/ with your own data
(or point SHEET_URLS to real Google Sheets) and delete storage/ to re-index.

    python scripts/make_sample_data.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR, SHEET_DIR = ROOT / "data" / "pdfs", ROOT / "data" / "sheets"
PDF_DIR.mkdir(parents=True, exist_ok=True)
SHEET_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# PDF content  ("H:" = heading, "P:" = paragraph, "BREAK" = new page)
# --------------------------------------------------------------------------
HANDBOOK = [
    "T:Delta Institute of Technology - Academic Handbook 2026/2027",
    "P:This handbook explains the academic regulations for undergraduate students of the "
    "Faculty of Computer Science and Artificial Intelligence. If there is a conflict, the "
    "decision of the Faculty Council prevails.",
    "H:1. Grading System",
    "P:Final course grades are calculated out of 100. Coursework counts for 40 percent of the "
    "course grade (midterm exam 20 percent, quizzes and assignments 20 percent) and the final "
    "examination counts for 60 percent.",
    "P:Grade A: 90 to 100 percent, 4.0 grade points. Grade B: 80 to 89 percent, 3.0 grade points. "
    "Grade C: 70 to 79 percent, 2.0 grade points. Grade D: 60 to 69 percent, 1.0 grade point. "
    "Grade F: below 60 percent, 0.0 grade points, and the course must be repeated.",
    "P:The Grade Point Average (GPA) is the sum of grade points multiplied by credit hours, divided "
    "by the total credit hours attempted in the semester. The Cumulative GPA (CGPA) covers all semesters.",
    "P:A student may repeat any course in which he or she earned grade D or F. The new grade replaces "
    "the old one in the CGPA, but a repeated course can earn a maximum grade of B.",
    "H:2. Course Registration",
    "P:The normal study load is between 12 and 18 credit hours per semester. Students with a semester "
    "GPA of 3.0 or higher in the previous semester may register up to 21 credit hours. Students on "
    "academic probation may register at most 12 credit hours. In the summer semester the maximum load "
    "is 9 credit hours.",
    "P:Students may add or drop courses during the first two weeks of the semester without any record. "
    "Late registration after this period requires approval from the academic advisor and a late "
    "registration fine of 300 EGP.",
    "P:A student may withdraw from a course until the end of week 10 and receive the grade W "
    "(Withdrawn), which does not affect the GPA. Withdrawal after week 10 results in grade F.",
    "P:A student cannot register for a course before passing its prerequisites, as listed in the "
    "course catalog.",
    "H:3. Attendance and Examinations",
    "P:Attendance is compulsory. A student who is absent from more than 25 percent of the sessions of "
    "a course without an accepted excuse is Deprived (grade DN) from that course and is not allowed "
    "to sit the final examination.",
    "P:Medical excuses must be submitted to the Student Affairs office within 5 working days of the "
    "absence, together with an official medical report.",
    "P:Midterm exams are held in week 8. Final examinations are held in weeks 15 and 16 of each semester.",
    "P:A student who misses an exam with an accepted excuse may sit a makeup exam after paying the "
    "makeup exam fee of 200 EGP per course.",
    "P:Any form of cheating results in grade F in the course and referral to the disciplinary committee.",
    "BREAK",
    "H:4. Grade Appeals",
    "P:A student may submit a grade appeal to the Student Affairs office within 7 days of the "
    "publication of results. The appeal committee announces its decision within 14 days. No fee is "
    "charged for appeals.",
    "H:5. Academic Standing",
    "P:A student whose CGPA falls below 2.0 at the end of a semester is placed on academic probation. "
    "A student who remains on probation for 3 consecutive semesters is dismissed from the program.",
    "P:The Dean's List includes students with a semester GPA of 3.6 or higher who registered for at "
    "least 15 credit hours and have no grade below C.",
    "BREAK",
    "H:6. Graduation Requirements",
    "P:To graduate, a student must complete 132 credit hours, achieve a CGPA of at least 2.0, pass the "
    "Graduation Project (CS490 and CS495) and complete the mandatory summer internship of 160 training "
    "hours.",
    "P:Graduation Project I can be registered only after completing 100 credit hours. The maximum "
    "duration of study is 7 years from the date of enrollment. Graduates with a CGPA of 3.5 or higher "
    "receive First Class Honors.",
]

SERVICES = [
    "T:Delta Institute of Technology - Student Services Guide",
    "H:1. Library",
    "P:The central library is open Sunday to Thursday from 8:00 to 20:00 and on Saturday from 9:00 to "
    "15:00. The library is closed on Friday. Undergraduate students may borrow up to 5 books for 14 "
    "days and may renew each loan twice. The late return fine is 5 EGP per book per day.",
    "H:2. Scholarships and Financial Aid",
    "P:Excellence Scholarship: covers 50 percent of tuition for the top 5 percent of each cohort with a "
    "CGPA of at least 3.8. Sports Scholarship: covers 25 percent of tuition for students who represent "
    "the institute in national competitions.",
    "P:Financial Aid: students with documented financial hardship may apply to the Financial Aid "
    "Committee through the Student Affairs office before the end of the second week of the semester. "
    "Scholarships do not cover lab fees or registration fees.",
    "H:3. Tuition Payment",
    "P:Tuition can be paid in 3 installments: 50 percent at registration, 25 percent in week 4 and 25 "
    "percent in week 8. Students who miss an installment deadline cannot access the learning platform "
    "Delta Learn until the payment is completed.",
    "BREAK",
    "H:4. IT Services",
    "P:Every student receives an institute email account and access to the Delta Learn learning "
    "management system. The IT Helpdesk is located in Building B, room 12, and is open Sunday to "
    "Thursday from 8:00 to 16:00. Contact: helpdesk@delta-institute.example or extension 1100. "
    "The campus Wi-Fi network is called Delta-Campus. Passwords can be reset at the helpdesk with a "
    "student ID.",
    "H:5. Career Center and Internships",
    "P:The Career Center organizes the mandatory summer internship of 160 training hours. Students "
    "become eligible for the internship after completing 60 credit hours. A career fair is held every "
    "March. The Career Center is in Building A, room 210.",
    "H:6. Health, Transport and Student Affairs",
    "P:The university clinic is free for all students and is open Sunday to Thursday from 9:00 to 15:00. "
    "University buses run from the main city bus station to the campus every 30 minutes between 7:00 "
    "and 18:00.",
    "P:The Student Affairs office is in Building A, room 105, extension 1050, and is open Sunday to "
    "Thursday from 9:00 to 14:00.",
]


def build_pdf(path: Path, content: list[str]) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=18, spaceAfter=14)
    head = ParagraphStyle("h", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=11, leading=15, spaceAfter=6)
    story = []
    for item in content:
        if item == "BREAK":
            story.append(PageBreak())
            continue
        kind, text = item.split(":", 1)
        story.append(Paragraph(text, {"T": title, "H": head, "P": body}[kind]))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      title=path.stem).build(story)


# --------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------
COURSES_HEADER = ["Course Code", "Course Name", "Credit Hours", "Level", "Semester",
                  "Prerequisites", "Instructor", "Day and Time", "Room", "Capacity"]
COURSES = [
    ["CS101", "Introduction to Programming", 3, 1, "Fall", "None", "Dr. Mona Fathy", "Sun 09:00-11:00", "Lab 1", 40],
    ["CS102", "Object-Oriented Programming", 3, 1, "Spring", "CS101", "Dr. Mona Fathy", "Mon 11:00-13:00", "Lab 2", 40],
    ["MA101", "Calculus I", 3, 1, "Fall", "None", "Dr. Hassan Ali", "Tue 09:00-11:00", "Hall A", 80],
    ["HU101", "Technical Writing", 2, 1, "Spring", "None", "Dr. Nadia Farouk", "Sat 10:00-12:00", "Hall D", 90],
    ["MA201", "Linear Algebra", 3, 2, "Fall", "MA101", "Dr. Hassan Ali", "Wed 11:00-13:00", "Hall B", 60],
    ["CS201", "Data Structures", 4, 2, "Fall", "CS102", "Dr. Karim Nabil", "Sun 11:00-14:00", "Lab 3", 35],
    ["CS210", "Database Systems", 3, 2, "Spring", "CS201", "Dr. Laila Samir", "Thu 09:00-12:00", "Lab 2", 35],
    ["ST201", "Probability and Statistics", 3, 2, "Spring", "MA101", "Dr. Hassan Ali", "Tue 11:00-14:00", "Hall A", 70],
    ["CS301", "Algorithms", 3, 3, "Fall", "CS201 and MA201", "Dr. Karim Nabil", "Mon 13:00-16:00", "Hall C", 50],
    ["CS310", "Operating Systems", 3, 3, "Spring", "CS201", "Dr. Omar Zaki", "Tue 12:00-15:00", "Hall B", 50],
    ["CS320", "Computer Networks", 3, 3, "Spring", "CS201", "Dr. Omar Zaki", "Wed 13:00-16:00", "Hall C", 50],
    ["CS330", "Software Engineering", 3, 3, "Fall", "CS210", "Dr. Laila Samir", "Tue 13:00-16:00", "Hall A", 50],
    ["AI301", "Introduction to Artificial Intelligence", 3, 3, "Fall", "CS201 and MA201", "Dr. Sara Khalil", "Wed 09:00-12:00", "Lab 4", 30],
    ["AI350", "Machine Learning", 4, 3, "Spring", "AI301", "Dr. Sara Khalil", "Sun 13:00-17:00", "Lab 4", 30],
    ["AI401", "Natural Language Processing", 3, 4, "Fall", "AI350", "Dr. Youssef Adel", "Mon 09:00-12:00", "Lab 5", 25],
    ["AI410", "Deep Learning", 3, 4, "Spring", "AI350", "Dr. Youssef Adel", "Thu 13:00-16:00", "Lab 5", 25],
    ["CS490", "Graduation Project I", 3, 4, "Fall", "100 completed credit hours", "Dr. Karim Nabil", "By appointment", "Project Room", 20],
    ["CS495", "Graduation Project II", 3, 4, "Spring", "CS490", "Dr. Karim Nabil", "By appointment", "Project Room", 20],
]
FEES_HEADER = ["Fee Item", "Amount EGP", "Applies To", "Notes"]
FEES = [
    ["Tuition per credit hour", 1800, "Computer Science program", "Paid in 3 installments"],
    ["Tuition per credit hour", 2200, "Artificial Intelligence program", "Paid in 3 installments"],
    ["Registration fee per semester", 1500, "All students", "Not covered by scholarships"],
    ["Lab fee per lab course", 600, "Courses taught in labs", "Not covered by scholarships"],
    ["Makeup exam fee per course", 200, "Students with an accepted excuse", "Per missed exam"],
    ["Late registration fine", 300, "Registration after week 2", "Needs advisor approval"],
    ["Graduation fee", 2500, "Graduating students", "Paid once"],
    ["Official transcript", 150, "All students", "Ready within 3 working days"],
    ["Student ID replacement", 100, "All students", "Lost or damaged card"],
]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    build_pdf(PDF_DIR / "academic_handbook.pdf", HANDBOOK)
    build_pdf(PDF_DIR / "student_services.pdf", SERVICES)
    write_csv(SHEET_DIR / "courses.csv", COURSES_HEADER, COURSES)
    write_csv(SHEET_DIR / "fees.csv", FEES_HEADER, FEES)
    print("Sample data written to", ROOT / "data")
