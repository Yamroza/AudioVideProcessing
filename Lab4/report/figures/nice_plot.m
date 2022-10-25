function f=plot_pv(x,y1,y2,y_label,output_filename)
      
    
    graph=plot(x,y1,'LineWidth',2);
    hold on;
    graph=scatter(x,y2,'LineWidth',2);
    %title(plot_title);
    grid on;
    xlabel("Epoch");
    ylabel(y_label);
   
    legend("Training","Validation");
    
    uistack(graph(1), 'bottom');
    set(gca,'FontSize',15);

    set(gcf, 'PaperUnits', 'centimeters ',  'OuterPosition', [150, 50, 1300, 850], 'PaperType', 'A4', 'PaperOrientation', 'landscape');
    set(gcf,'Position', [200,100,1200,800]);
    print(gcf, '-dpdf','-loose','-opengl','-r600', strcat(output_filename, '.pdf'));
    close all;
end
