package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

public class Main extends Application
{


    @Override
    public void start(Stage primaryStage) throws Exception
    {
        Parent root = FXMLLoader.load(getClass().getResource("sample.fxml"));
        primaryStage.setTitle("Musiphile");
        primaryStage.setScene(new Scene(root, 600, 450));
        primaryStage.getIcons().add(new Image("file:D:\\Personal_Projects\\Java\\Javafx\\Musiphile\\src\\sample\\music.jpg"));
        primaryStage.show();
    }


    public static void main(String[] args)
    {
        launch(args);
    }
}