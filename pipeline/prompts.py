INSTRUCTION = '''Solve a question answering task with interleaving Thought, Action, Answer steps. Thought can reason about the current situation, and Action can be three types: 
(1) Question[question], which returns a sub-question with single answer that helps solve the original problem.
(2) Multi_Answer_Question[keyword], which returns a sub-question with multiple answers.
(3) Finish[answer], which returns the answer and finishes the task.
Here is an example.\n'''

CWQ_EXAMPLE = '''\nQuestion Who are the children of Ike and the woman who composed Nutbush City Limits?
Thought 1 I need to find out who Ike is and who composed Nutbush City Limits, and then determine their children.
Action 1 Question[Who is Ike?] 
Answer 1 Ike Turner
Thought 2 Ike Turner is the person being referred to. I now need to find out who composed Nutbush City Limits. 
Action 2 Question[Who composed Nutbush City Limits?] 
Answer 2 Tina Turner
Thought 3 Tina Turner composed Nutbush City Limits. I now need to find out who the children of Ike and Tina Turner are.
Action 3 Multi_Answer_Question[Who are the children of Ike and Tina Turner?] 
Answer 3 Ronnie Turner and Raymond Craig Turner.
Thought 4 I found the children of Ike and Tina Turner: Ronnie Turner and Raymond Craig Turner.
Action 4 Finish[Ronnie Turner and Raymond Craig Turner]\n'''

WEQSP_EXAMPLE = '''\nQuestion Who are the children of Ike and the woman who composed Nutbush City Limits?
Thought 1 I need to find out who Ike is and who composed Nutbush City Limits, and then determine their children.
Action 1 Question[Who is Ike?] 
Answer 1 Ike Turner
Thought 2 Ike Turner is the person being referred to. I now need to find out who composed Nutbush City Limits. 
Action 2 Question[Who composed Nutbush City Limits?] 
Answer 2 Tina Turner
Thought 3 Tina Turner composed Nutbush City Limits. I now need to find out who the children of Ike and Tina Turner are.
Action 3 Multi_Answer_Question[Who are the children of Ike and Tina Turner?] 
Answer 3 Ronnie Turner and Raymond Craig Turner.
Thought 4 I found the children of Ike and Tina Turner: Ronnie Turner and Raymond Craig Turner.
Action 4 Finish[Ronnie Turner and Raymond Craig Turner]\n'''

INSTRUCTION_COT = '''Solve a question by giving your thoughts before giving the answer. Give the answer after the phrase \"So the answer is \".
Here is an example.\n'''

CWQ_EXAMPLE_COT = '''\nQ: Who are the children of Ike and the woman who composed Nutbush City Limits?
A: The woman who composed Nutbush City Limits is Tina Turner, and the children of Ike and Tina Turner are Ronnie Turner and Raymond Craig Turner. So the answer is Ronnie Turner and Raymond Craig Turner.\n'''

WEBQSP_EXAMPLE_COT = '''\nQ: What character did john noble play in lord of the rings?
A: John Noble is a well-known actor who has played many roles in movies and TV shows. In the Lord of the Rings movie trilogy, he played a character who was an important figure in the city of Gondor. He was a wise and powerful leader, but also had a tragic story arc. So the answer is Denethor II.\n'''

INSTRUCTION_DIRECT = '''Solve a question by giving the answer after the phrase \"A: \".
Here are some examples.\n'''

CWQ_EXAMPLE_DIRECT = '''\nQ: Who are the children of Ike and the woman who composed Nutbush City Limits?
A: Ronnie Turner and Raymond Craig Turner.

Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: Washington.

Q: In what year did the basketball team whose head coach is Kevin McHale win their first championship game?
A: 1994 NBA Finals.

Q: What NCAA football team won the 1993 College World Series championship?
A: Louisiana State Fighting Tigers Mens Football.\n'''

WEBQSP_EXAMPLE_DIRECT = '''\nQ: Which asian country has the biggest population?
A: India.

Q: What movies did james franco play in?
A: Wild Horses.

Q: What does bob dylan sing?
A: Hurricane.

Q: Where does the appalachian trail run through?
A: Eastern United States.\n'''