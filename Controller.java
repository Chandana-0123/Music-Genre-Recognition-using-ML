package sample;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import java.lang.*;
import com.gluonhq.charm.glisten.control.TextField;
import java.io.*;


public class Controller
{

    @FXML
    TextArea Output_Display;
    @FXML
    CheckBox Checker;
    @FXML
    CheckBox Checker1;
    @FXML
    TextField Input_Path;
    @FXML
    CheckBox Checker2;


    private boolean exit;

    String Output;
    String Error;

    public void Run()
    {

        int validator = 0;
        int validator1 = 0;
        int validator2 = 0;

        try
        {
            if (Checker.isSelected())
            {
                validator = 1;
            }
            if (Checker1.isSelected())
            {
                validator1 = 1;
            }
            if (Checker2.isSelected())
            {
                validator2 = 1;
            }
            Process process = Runtime.getRuntime().exec("python -u D:\\Personal_Projects\\Python\\Musiphile\\MusiphileVariation1.py " + Input_Path.getText() + " " + validator + " " + validator1 + " " + validator2);
            BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));

            while ((Output = stdInput.readLine()) != null)
            {
                Output_Display.setText("Song is Classified as " + Output);
            }

        }
        catch (IOException exception)
        {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

     public void web_search()
     {
         try
         {
             String Search = "Start https://www.google.com/search?q=" + Output + " songs";
             Process process1 = Runtime.getRuntime().exec("cmd /c " + Search);
         }
         catch (Exception exception)
         {
             Platform.runLater(() -> Output_Display.setText("Exception Encountered!!!"));
             exception.printStackTrace();
             System.exit(-1);
         }
     }

     public void reset()
     {
         try
         {
             Platform.runLater(() ->Input_Path.setText(""));
         }
         catch(Exception exception)
         {
             Platform.runLater(() -> Output_Display.setText("Exception Encountered!!!"));
             exception.printStackTrace();
             System.exit(-1);
         }
     }
    public void exit()
    {

        try
        {
            System.exit(0);
        }
        catch(Exception exception)
        {
            Platform.runLater(() -> Output_Display.setText("Exception Encountered!!!"));
            exception.printStackTrace();
            System.exit(-1);
        }
    }
}
//D:\\Personal_Projects\\Python\\Musiphile\\Audio\\AudioRockOriginal.mp3

