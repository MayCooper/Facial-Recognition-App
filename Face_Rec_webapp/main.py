from flask import Flask
import sys
sys.path.insert(0, '')  # The empty string represents the current directory.

from my_app import views



app = Flask(__name__)

# url
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/faceapp','faceapp',views.faceapp)
app.add_url_rule('/faceapp/gender','gender',views.gender,methods=['GET','POST'])
# 
if __name__ == "__main__":
    app.run(debug=True)