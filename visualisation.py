# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:36:41 2021

@author: dia
"""

import sys
import os
from pyvis.network import Network
from bs4 import BeautifulSoup
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLineEdit, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QPixmap
import matplotlib.image as mpimg
from PyQt5.QtGui import QImage, QPixmap
import cv2
from UI_MainWindow import Ui_MainWindow
from UI_GraphWindow import Ui_GraphWindow
from Modele import *
from PandasDFModel import DataFrameModel

mainDir = os.path.dirname(os.path.realpath(__file__))

#cette classe gère toutes les interactions avec le menu principal
class MainWindow(QMainWindow):
    def __init__(self, graphWindow, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.corpus = Corpus("Corona")
        self.graphWindow = graphWindow
        self.ui.DfButton.clicked.connect(self.PreviewResult)
        self.ui.GraphButton.clicked.connect(self.ShowGraph)
        self.ui.CentralityButton.clicked.connect(self.Centrality_graph)

        

    #cette fonction fait appel au foncton suivant ce que l'utilisateur a coché
    def ExecuteFonction(self):
        
        singleElement = self.ui.SingleSelectorInput_2.text()
        
        if self.ui.LimitResultCheck_3.isChecked():
            self.corpus.N = self.ui.MaxResultSelector_3.value()
        else:
            self.corpus.N  = 15
        
        
        if self.ui.RButtonMotifSelect.isChecked():
            self.ui.SingleSelectorInput_2.setEnabled(True)
            self.corpus.co_occurenceMotif(singleElement)
            self.corpus.centralityMotif(singleElement)
            return True
        elif self.ui.RButtonIdSelect_3.isChecked():
            self.corpus.co_occurence()
            self.corpus.centrality()
            return True
        
        
    #cette fonction permet de prévisualiser le dataframes des co-occurences 
    def PreviewResult(self):
        isValid = self.ExecuteFonction()
        if isValid:
            model = DataFrameModel(self.corpus.result_df)
            self.ui.PreviewDF.setModel(model)
        
    #cette focntion gère l'affichage du graphe
    def ShowGraph(self):
        isValid = self.ExecuteFonction()
        if isValid:
            self.graphWindow.graphManager.GenerateGraph(self.corpus.result)
            self.graphWindow.show()
    
    #cette fonction permet de fermer la fenêtre du graphe
    def closeEvent(self, event):
        self.graphWindow.close()   
        
    #https://stackoverflow.com/questions/54735982/how-to-load-an-image-in-label-when-button-pressed-pyqt5 ce lien nous a aidé pour afficher et charger l'image
     #cette fonction gère l'affichage de l'image
    def displayImage(self):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

                img = QImage(self.image,
                             self.image.shape[1],
                             self.image.shape[0],
                             self.image.strides[0],
                             qformat)
                img = img.rgbSwapped()

                self.ui.PhotoGraph.setPixmap(QPixmap.fromImage(img))      
                self.ui.PhotoGraph.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                
    #cette fonction gère le chargement de  l'image
    def loadImage(self, flname, cv ):                 
        self.image = cv2.imread(flname)
        self.displayImage()
        
    #cette focntion affiche les centralités des graphes
    def Centrality_graph(self):
         
         isValid = self.ExecuteFonction()
         if isValid:
            self.loadImage("Graph.png",cv2.IMREAD_GRAYSCALE)
         
        
        
#cette classe gère les interactions avec la fen^tre du graphe       
class GraphWindow(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super(GraphWindow, self).__init__(*args, **kwargs)
        self.ui = Ui_GraphWindow()
        self.ui.setupUi(self)
        self.webview = QWebEngineView()
        self.ui.GraphLayout.addWidget(self.webview)
        self.graphManager = GraphManager(self.webview)
        self.ui.actionCapturer.triggered.connect(self.SaveGraph)
        
    
    #cette fonctonpermet de saugarder le graphe
    def SaveGraph(self):
        self.webview.grab().save("graph.png")
        msg = QMessageBox()
        msg.setWindowTitle("Info")
        msg.setText("Graph exporté avec succès")
        msg.exec_() 
        
        
        
    
#cette classe gère le chargement du graphe       
class GraphManager:
    def __init__(self, webview):
        self.webview = webview
        self.graph = Network()
        self.graph.set_options('var options = {"autoResize": true, "height": "100%", "width": "100%", "locale": "fr", "clickToUse": false}')
        
        self.__custom_style__ = """
            body{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #white}
            h1{visibility: collapse;}
            #mynetwork {
                width: 100%;
                height: 100%;
                background-color: #white;
                border: 0;
                position: absolute;
                top: 0;
                left: 0;}
        """
    #fonction qui génère le graphe
    def GenerateGraph(self, data):
        self.graph = data
        self.graph.barnes_hut(gravity=-40000, central_gravity=0.3, spring_length=250, spring_strength=0.01, damping=0.09, overlap=0)
        self.graph.save_graph("graph.html")
        graph_path = os.path.abspath(os.path.join(mainDir, "graph.html"))
        
        soup = BeautifulSoup(open("graph.html").read(),features="html.parser")
        style_tag = soup.find("style")
        style_tag.string = self.__custom_style__
        open("graph.html", "w", encoding="utf-8").write(str(soup))
        
        self.webview.load(QUrl.fromLocalFile(graph_path))
        


if __name__ == '__main__':
    
    #SETUP APP
    app = QApplication(sys.argv)
    
   
    
    #SETUP WINDOWS
    graph_win = GraphWindow()
    Main_win = MainWindow(graph_win)
    
    #DISPLAY WINDOWS
    Main_win.show()
    
    app.exec_()
    sys.exit()
    
        
