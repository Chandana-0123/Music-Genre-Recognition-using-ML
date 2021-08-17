package sample;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import java.lang.*;
import com.gluonhq.charm.glisten.control.TextField;
import java.io.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.util.Duration;

public class Controller {

    @FXML
    private TextArea Output_Display, About;
    @FXML
    private CheckBox Checker, Checker1, Checker2, Checker3;
    @FXML
    private TextField Input_Path;
    @FXML
    private MediaView MV;
    @FXML
    private Slider VC,PC;
    @FXML
    private Button OK;
    @FXML
    private ToggleButton repeater;
    @FXML
    private WebView Browse;
    @FXML
    private ImageView animate, musicanimate;


    private boolean exit;
    MediaPlayer MP;
    Duration a;
    String Output = null;
    int status = 0;
    int init;
    String Passer;

    public void Run() {
        try {
                int validator = 0;
                int validator1 = 0;
                int validator2 = 0;
                int validator3 = 0;
                init = 0;
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
                if (Checker3.isSelected())
                {
                    validator3 = 1;
                }
                Process process = Runtime.getRuntime().exec("python -u D:\\Personal_Projects\\Python\\Musiphile\\Code\\Musiphile.py " + Input_Path.getText() + " " + validator + " " + validator1 + " " + validator2 + " " + validator3);
                BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
                BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                Output_Display.appendText("Song is Classified as ");
                while ((Output = stdInput.readLine()) != null)
                {
                    Output_Display.appendText(Output + "\n");
                    if(init == 0)
                    {
                        Passer = Output;
                        init = 1;
                    }
                }
                if (Passer.equals("Jazz"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Jazz.gif"));
                }
                else if (Passer.equals("Rock"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Rock.gif"));
                }
                else if (Passer.equals("Metal"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Metal.gif"));
                }
                else if (Passer.equals("Reggae"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Reggae.gif"));
                }
                else if (Passer.equals("Blues"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Blues.gif"));
                }
                else if (Passer.equals("Hip-Hop"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\HIPHOP.gif"));
                }
                else if (Passer.equals("Classical"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Classical.gif"));
                }
                else if (Passer.equals("Country"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Country.gif"));
                }
                else if (Passer.equals("Pop"))
                {
                    animate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\Pop.gif"));
                }

        }
        catch (Exception exception)
        {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    public void web_search() {
        try
        {
            WebEngine WE = Browse.getEngine();
            String Search = "https://www.google.com/search?q=" + Passer + "+songs";
            Browse.setZoom(0.75);
            WE.load(Search);
        }
        catch (Exception exception)
        {
            Platform.runLater(() -> Output_Display.setText("Exception Encountered!!!"));
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    public void reset() {
        try
        {
            Platform.runLater(() -> Input_Path.setText(""));
        }
        catch (Exception exception)
        {
            Platform.runLater(() -> Output_Display.setText("Exception Encountered!!!"));
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    public void exit() {

        try
        {
            System.exit(0);
        }
        catch (Exception exception) {
            Platform.runLater(() -> Output_Display.setText("Exception Encountered!!!"));
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    public void play()
    {
        File file = new File(Input_Path.getText());
        Media media = new Media(file.toURI().toString());
        MP = new MediaPlayer(media);
        MV.setMediaPlayer(MP);
        VC.setValue(MP.getVolume()*100);
        VC.valueProperty().addListener(observable -> MP.setVolume(VC.getValue()/100));
        PC.setValue(MP.getRate()*50);
        PC.valueProperty().addListener(observable -> MP.setRate(PC.getValue()/50));
        if(status == 0)
        {
            MP.play();
            status = 1;
            musicanimate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\sample\\Play.gif"));
        }
        if(status == 1)
        {
            MP.stop();
            musicanimate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\sample\\StillPlay.jpg"));
            MP.play();
            musicanimate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\sample\\Play.gif"));
        }
        if(status == 2)
        {
            MP.setStartTime(a);
            MP.play();
            musicanimate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\sample\\Play.gif"));
            status = 1;
        }
    }

    public void pause()
    {
        a = MP.getCurrentTime();
        MP.pause();
        musicanimate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\sample\\StillPlay.jpg"));
        status = 2;
    }
    public void stop()
    {
        MP.stop();
        musicanimate.setImage(new Image("file:D:\\Personal_Projects\\Java\\JavaFX\\Musiphile\\src\\sample\\StillPlay.jpg"));
        status = 0;
    }

    public void Display()
    {
        About.setStyle("-fx-opacity: 1.0");
        OK.setStyle("-fx-opacity: 1.0");
    }


    public void hide()
    {
        About.setStyle("-fx-opacity: 0.0");
        OK.setStyle("-fx-opacity: 0.0");
    }

    public void repeat()
    {
        if(repeater.isSelected())
        {
            MP.setOnEndOfMedia(new Runnable() {
                @Override
                public void run() {
                    MP.seek(Duration.ZERO);
                    MP.stop();
                }
            });
            MP.play();
        }
    }
}
//D:\\Personal_Projects\\Python\\Musiphile\\Audio\\
