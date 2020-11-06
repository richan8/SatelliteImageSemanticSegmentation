var locLat = 40.6110;
var locLong = -74.0299;
var map1;
var map2;
var zoomLevel = 18
var runMapBool = false

const sleep = (milliseconds) => {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

function runMap(dLat, dLong,time){
  moveMaps(dLat, dLong);
  sleep(time).then(() => {
    if (runMapBool) {
      runMap(dLat, dLong, time);
    }
  });
}

function keyPress(x){
  var x = x.which || x.keyCode;
  if(x == 32){
    if(!runMapBool){
      runMapBool = true;
      map1.setZoom(zoomLevel);
      map2.setZoom(zoomLevel);
      locLat = parseFloat(map1.center['lat']());
      locLong = parseFloat(map1.center['lng']());
      runMap(0.0004,0,100); ///////////// RUN DIRECTION
    }
    else{
      runMapBool=false;
    }
  }
}

function moveMaps(dLat,dLong){
  locLat = locLat+dLat
  locLong = locLong+dLong
  var newLoc = new google.maps.LatLng({lat: locLat, lng: locLong});
  map1.panTo(newLoc);
  map2.panTo(newLoc);
}

function initMaps() {
  map1 = new google.maps.Map(document.getElementById("map1"), {
    center: { lat: locLat, lng: locLong },
    zoom: zoomLevel,
    disableDefaultUI: true,
    mapTypeId: 'satellite'
  });
  map2 = new google.maps.Map(document.getElementById("map2"), {
    center: { lat: locLat, lng: locLong },
    zoom: zoomLevel,
    disableDefaultUI: true,
    styles: [
      {elementType: 'geometry.stroke', stylers: [{ visibility: "off" }]},
      {elementType: 'labels.text.stroke', stylers: [{ visibility: "off" }]},
      {elementType: 'labels.text.fill', stylers: [{ visibility: "off" }]},
      {
        featureType: 'administrative',
        elementType: 'labels',
        stylers: [{ visibility: "off" }]
      },{
        featureType: 'poi',
        elementType: 'labels',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'poi',
        elementType: 'geometry',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'poi.park',
        elementType: 'geometry',
        stylers: [{visibility: "off"}]
      },
      {
        featureType: 'poi.park',
        elementType: 'labels',
        stylers: [{visibility: "off"}]
      },
      {
        featureType: 'road',
        elementType: 'geometry',
        stylers: [{color: '#000000'}]
      },
      {
        featureType: 'road',
        elementType: 'geometry.stroke',
        stylers: [{color: '#000000'}]
      },
      {
        featureType: 'road',
        elementType: 'labels.text.fill',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'road',
        elementType: 'labels.icon',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'transit',
        elementType: 'geometry',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'transit',
        elementType: 'labels',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'water',
        elementType: 'geometry',
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: 'water',
        elementType: 'labels',
        stylers: [{ visibility: "off" }]
      }
    ]
  });

  map1.setTilt(0);
  map2.setTilt(0);
}

$(document).ready(function(){
  console.log('Press space to start');
});