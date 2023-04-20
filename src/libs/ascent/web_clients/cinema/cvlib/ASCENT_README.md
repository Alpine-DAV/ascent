2019/06/18 Adapted from:

6ff935e (HEAD -> master, origin/master, origin/HEAD) HEAD@{0}: clone: from https://github.com/cinemascience/cvlib.git

One change, which allows the html client page to exist right beside the info.json:


diff --git a/js/DatabaseSpecA.js b/js/DatabaseSpecA.js
index fd1ee51..e1af7f5 100644
--- a/js/DatabaseSpecA.js
+++ b/js/DatabaseSpecA.js
@@ -18,7 +18,10 @@ function DatabaseSpecA(url, callback){
         dataType: 'json',
         url: url,
         success: function(json){
-            json.databaseDir = prefix+'/';
+            if( prefix === "")
+                json.databaseDir = '';
+            else
+                json.databaseDir = prefix+'/';
             if(!self.validateJSON(json))
                 return;
             self.json = json;
