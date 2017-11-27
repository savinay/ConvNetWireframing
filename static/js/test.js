var circle = new Path.Circle({
	center: [80, 50],
	radius: 5,
	fillColor: 'red'
});

// Create a rasterized version of the path:
var raster = circle.rasterize();

// Move it 100pt to the right:
raster.position.x += 100;

// Scale the path and the raster by 300%, so we can compare them:
circle.scale(5);
raster.scale(5);