# from sqlalchemy import create_engine, text
# 
# engine = create_engine("postgresql+psycopg2://postgres:12345678@localhost:5432/student_model")
# 
# with engine.connect() as conn:
#     result = conn.execute(text("SELECT version();"))
#     print(result.fetchone())


import psycopg2

# Step 1: Connect to PostgreSQL
connection = psycopg2.connect(
    database="student_model",   # replace with your db name
    user="postgres",   # your postgres username
    password="12345678",  # your postgres password
    host="localhost",
    port="5432"
)

# Step 2: Create a cursor object
cursor = connection.cursor()

# Step 3: Create a table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(25),
    class VARCHAR(25),
    Section VARCHAR(25),
    Marks int
);
""")

# Step 4: Insert a record
cursor.execute('''INSERT INTO students (name, class,Section,Marks) VALUES ('Alok','MCA','A',450)''')
cursor.execute('''INSERT INTO students (name, class,Section,Marks) VALUES ('Rahul','Btech','B',400)''')
cursor.execute('''INSERT INTO students (name, class,Section,Marks) VALUES ('Rohit','MBA','C',350)''')
cursor.execute('''INSERT INTO students (name, class,Section,Marks) VALUES ('Sahil','MBA','A',300)''')
cursor.execute('''INSERT INTO students (name, class,Section,Marks) VALUES ('Aman','MCA','B',250)''')

# Step 5: Commit changes
# connection.commit()

# Step 6: Query the table
cursor.execute("SELECT * FROM students")
data = cursor.fetchall()
for row in data:
    print(row)

# Commit changes
connection.commit()


# Step 7: Close connection
cursor.close()
connection.close()


