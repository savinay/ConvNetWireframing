console.log("app.js");

$("#recognize").click(function() {
        console.log("recognize");
        // domtoimage.toBlob(document.getElementById('canvas'))
        // .then(function (blob) {
        //     console.log(blob);
        //     window.saveAs(blob, 'canvas.png');
        // })
        var png = ReImg.fromCanvas(document.getElementById('canvas')).toPng();
        ReImg.fromCanvas(document.querySelector('canvas')).downloadPng();
        <!-- console.log(png); -->
        $.ajax({
          type: "POST",
          url: "/predict",
          data: "canvas",
          success: function(obj) {
            console.log(obj);
            console.log(success);
          },
          error: function(err) {
            console.log(err);
          }
        });
        
    })