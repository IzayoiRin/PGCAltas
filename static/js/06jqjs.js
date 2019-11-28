$(function(){
    $('.buts button').eq(0).click(function(){
        $(this).css('background', 'lightblue').siblings().toggle(200)
        $(this).parent().toggleClass('buts_c')
        $('.content').toggle()
    })

    $('.buts button').not('.buts button:eq(0)').mouseover(function(){
        $(this).css('background', 'lightblue').siblings().css('background', '')
        $('.content').children().eq($(this).index()-1).addClass('content_c').siblings().removeClass('content_c')
    })

    $('.cas li').mouseover(function(){
        $(this).children('ul').stop().show(200)
    })

    $('.cas li').mouseout(function(){
        $(this).children('ul').stop().hide(200)
    })
})